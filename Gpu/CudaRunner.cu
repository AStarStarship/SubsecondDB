// Copyright AStarship <https://astarship.net>.
//
// SubsecondDb GPU Vector Database — CUDA runner implementation.
//
// BARE-METAL <cuda_runtime.h>. Compiled by nvcc only (see CMakeLists.txt);
// the CPU build never sees this file. The kernels reproduce
// VectorIndex::Score exactly so the GPU ranking is cross-checkable against
// the CPU reference.

#include "CudaRunner.h"

#include <cuda_runtime.h>

namespace _ {

namespace {

// A (score, id, slot) record for the phase-2 top-k selection. FPD score so
// the accumulation matches the CPU reference; IUD id + IUC slot.
struct TMeta {
  FPD  score;
  IUD  id;
  IUC  slot;
};

// ---------------------------------------------------------------------------
// Phase 1: one thread scores one vector. Grid-stride over [0, count).
// metric: 0 = L2Squared (sum of squared diffs), 1 = InnerProduct (neg dot).
// Writes scores to d_scores, (id,slot) to d_meta.
// ---------------------------------------------------------------------------
__global__ void KScore(const float* __restrict__ query,
                       const float* __restrict__ data,
                       const unsigned long long* __restrict__ ids,
                       double* __restrict__ scores,
                       unsigned long long* __restrict__ ids_out,
                       unsigned int* __restrict__ slots_out,
                       unsigned int dim, unsigned int count,
                       unsigned int metric) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int stride = gridDim.x * blockDim.x;
  for (; i < count; i += stride) {
    const float* row = data + (size_t)i * dim;
    double acc = 0.0;
    if (metric == 0) {  // L2Squared
      for (unsigned int d = 0; d < dim; ++d) {
        double diff = (double)query[d] - (double)row[d];
        acc += diff * diff;
      }
    } else {  // InnerProduct: negative dot
      for (unsigned int d = 0; d < dim; ++d) {
        acc += (double)query[d] * (double)row[d];
      }
      acc = -acc;
    }
    scores[i] = acc;
    ids_out[i] = ids[i];
    slots_out[i] = i;
  }
}

// ---------------------------------------------------------------------------
// Phase 2: single-block bounded max-heap top-k over the phase-1 scratch.
// K <= blockDim.x (we launch one block of K threads... actually one block of
// up to 1024 threads; K is small so we use the first K threads as heap
// slots and the rest as producers). Simpler + correct: one thread per
// candidate, insert into a shared max-heap of size K with atomic-free
// serialization via __syncthreads between "insert" phases is overkill.
//
// Correct approach without atomics: each of the first K threads owns one
// heap slot; every thread proposes its (score,id,slot); we do a serial
// bounded-heap insert guarded by a single "heap steward" thread (thread 0)
// after a __syncthreads that stages all candidates. For a flat index the
// candidate set is `count`, which can be large, so we do a grid-stride where
// EACH BLOCK keeps a local max-heap of size K, then a second kernel merges
// the per-block K-best. That is the standard two-stage top-k and avoids any
// cross-block synchronization.
//
// This kernel = stage 2a: each block reduces its slice to its local K-best
// (max-heap in shared mem, root = worst-of-K), writing K (score,id,slot)
// per block to d_block_best (gridDim.x * K records).
// ---------------------------------------------------------------------------
__global__ void KBlockTopK(const double* __restrict__ scores,
                           const unsigned long long* __restrict__ ids,
                           const unsigned int* __restrict__ slots,
                           TMeta* __restrict__ block_best,
                           unsigned int dim, unsigned int count,
                           unsigned int K) {
  __shared__ TMeta heap[1024];   // K <= 1024; K is the top-k bound
  __shared__ int   heap_n;

  if (threadIdx.x == 0) heap_n = 0;
  __syncthreads();

  // Stage this block's slice through a shared staging buffer. Threads write
  // their candidates into stage[] in parallel; a single __syncthreads per
  // staging pass (NOT per candidate) lets thread 0 drain them all into the
  // bounded max-heap. This batches the barrier: O(count/STAGE) syncs instead
  // of O(count). STAGE must be a multiple of blockDim.x and <= 1024.
  constexpr int STAGE = 256;
  __shared__ TMeta stage[STAGE];

  // Each block round-robins over STAGE-sized chunks: block b handles chunk b,
  // b + gridDim.x, b + 2*gridDim.x, ... so across the grid every index
  // element is staged exactly once.
  for (unsigned int base = blockIdx.x * blockDim.x; base < count;
       base += gridDim.x * STAGE) {
    // Fill the staging buffer for this pass (up to STAGE candidates).
    for (unsigned int off = threadIdx.x; off < STAGE; off += blockDim.x) {
      unsigned int i = base + off;
      if (i < count) {
        stage[off].score = scores[i];
        stage[off].id    = ids[i];
        stage[off].slot  = slots[i];
      } else {
        stage[off].score = 1e300;  // +inf: ignored by the heap
        stage[off].id = 0;
        stage[off].slot = 0;
      }
    }
    __syncthreads();  // one barrier per staging pass
    if (threadIdx.x == 0) {
      for (int s = 0; s < STAGE; ++s) {
        TMeta cand = stage[s];
        if (cand.score >= 1e299) continue;  // skip +inf padding
        int n = heap_n;
        if (n < K) {
          heap[n] = cand; ++n;
          int p = n - 1;
          while (p > 0) {
            int par = (p - 1) / 2;
            if (heap[par].score >= heap[p].score) break;
            TMeta t = heap[par]; heap[par] = heap[p]; heap[p] = t;
            p = par;
          }
        } else if (cand.score < heap[0].score) {
          heap[0] = cand;
          int p = 0;
          for (;;) {
            int l = 2 * p + 1, r = 2 * p + 2, m = p;
            if (l < K && heap[l].score > heap[m].score) m = l;
            if (r < K && heap[r].score > heap[m].score) m = r;
            if (m == p) break;
            TMeta t = heap[m]; heap[m] = heap[p]; heap[p] = t;
            p = m;
          }
        }
        heap_n = n;
      }
    }
    __syncthreads();  // before the next pass overwrites stage[]
  }

  // Stage this block's K-best (max-heap, root = worst) to global.
  if (threadIdx.x == 0) {
    int n = heap_n;
    for (int i = 0; i < n; ++i) {
      block_best[blockIdx.x * K + i] = heap[i];
    }
    // Pad the rest with +inf so the merge ignores them.
    for (int i = n; i < K; ++i) {
      TMeta inf; inf.score = 1e300; inf.id = 0; inf.slot = 0;
      block_best[blockIdx.x * K + i] = inf;
    }
  }
}

// ---------------------------------------------------------------------------
// Stage 2b: single-block merge of (gridDim.x * K) block-best into the global
// K-best, ascending by score (best first). One block, K <= 1024 threads not
// needed — thread 0 does the bounded-heap over gridDim.x*K candidates.
// ---------------------------------------------------------------------------
__global__ void KMergeTopK(const TMeta* __restrict__ block_best,
                           TMeta* __restrict__ out,
                           unsigned int blocks, unsigned int K) {
  // Total candidates = blocks * K.
  __shared__ TMeta heap[1024];
  __shared__ int   heap_n;
  if (threadIdx.x == 0) heap_n = 0;
  __syncthreads();
  if (threadIdx.x != 0) return;  // thread 0 does the serial bounded-heap

  const unsigned int total = blocks * K;
  for (unsigned int i = 0; i < total; ++i) {
    TMeta cand = block_best[i];
    int n = heap_n;
    if (cand.score >= 1e299) continue;  // skip +inf padding
    if (n < K) {
      heap[n] = cand; ++n;
      int p = n - 1;
      while (p > 0) {
        int par = (p - 1) / 2;
        if (heap[par].score >= heap[p].score) break;
        TMeta t = heap[par]; heap[par] = heap[p]; heap[p] = t;
        p = par;
      }
    } else if (cand.score < heap[0].score) {
      heap[0] = cand;
      int p = 0;
      for (;;) {
        int l = 2 * p + 1, r = 2 * p + 2, m = p;
        if (l < K && heap[l].score > heap[m].score) m = l;
        if (r < K && heap[r].score > heap[m].score) m = r;
        if (m == p) break;
        TMeta t = heap[m]; heap[m] = heap[p]; heap[p] = t;
        p = m;
      }
    }
    heap_n = n;
  }
  // heap is a max-heap (root = worst of the K). Drain descending, reverse.
  // Use heap_n as the FIXED loop bound; n is the shrinking live-size used for
  // the in-place pop. (Binding the loop to n would stop after ~K/2 pops.)
  __shared__ TMeta desc[1024];
  int n = heap_n;
  for (int i = 0; i < heap_n; ++i) {
    desc[i] = heap[0];
    TMeta last = heap[n - 1];
    --n;
    if (n > 0) {
      heap[0] = last;
      int p = 0;
      for (;;) {
        int l = 2 * p + 1, r = 2 * p + 2, m = p;
        if (l < n && heap[l].score > heap[m].score) m = l;
        if (r < n && heap[r].score > heap[m].score) m = r;
        if (m == p) break;
        TMeta t = heap[m]; heap[m] = heap[p]; heap[p] = t;
        p = m;
      }
    }
  }
  // desc[0..heap_n-1] is descending (worst first); write ascending (best first).
  for (int i = 0; i < heap_n; ++i) out[i] = desc[heap_n - 1 - i];
}

// Launch config helpers.
constexpr IUC kThreadsPerBlock = 256;
IUC GridForCount(IUC count) {
  IUC blocks = (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
  return blocks > 0 ? blocks : 1;
}

CudaStatus MapCudaError(cudaError_t e) {
  return e == cudaSuccess ? CudaStatus::Ok : CudaStatus::CudaError;
}

}  // namespace

// ---------------------------------------------------------------------------
CudaStatus CudaRunner::Init(ISN ordinal) {
  if (inited) return CudaStatus::Ok;  // idempotent
  int count_devices = 0;
  cudaError_t e = cudaGetDeviceCount(&count_devices);
  if (e != cudaSuccess) return CudaStatus::CudaError;
  if (count_devices == 0) return CudaStatus::NoDevice;

  ISN ord = (ordinal < 0) ? 0 : ordinal;
  if (ord >= count_devices) return CudaStatus::NoDevice;
  e = cudaSetDevice(ord);
  if (e != cudaSuccess) return CudaStatus::CudaError;
  device_ordinal = ord;

  cudaDeviceProp prop;
  e = cudaGetDeviceProperties(&prop, ord);
  if (e != cudaSuccess) return CudaStatus::CudaError;

  // Copy device name into the fixed buffer (no std::string).
  for (IUC i = 0; i < 64; ++i) device_name_[i] = 0;
  IUC len = 0;
  while (prop.name[len] && len < 63) { device_name_[len] = prop.name[len]; ++len; }
  device_name_[len] = 0;

  vram_total_mib = (IUC)(prop.totalGlobalMem / (1024 * 1024));
  size_t free_bytes = 0, total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  vram_free_mib = (IUC)(free_bytes / (1024 * 1024));
  compute_capability_major = (IUB)prop.major;
  compute_capability_minor = (IUB)prop.minor;
  inited = true;
  return CudaStatus::Ok;
}

// ---------------------------------------------------------------------------
CudaStatus CudaRunner::Shutdown() {
  if (d_data)    { cudaFree(d_data);    d_data = nullptr; }
  if (d_ids)     { cudaFree(d_ids);     d_ids = nullptr; }
  if (d_scores)  { cudaFree(d_scores);  d_scores = nullptr; }
  if (d_meta)    { cudaFree(d_meta);    d_meta = nullptr; }
  if (d_query)   { cudaFree(d_query);   d_query = nullptr; }
  count = 0;
  dim = 0;
  if (inited) cudaDeviceReset();
  inited = false;
  device_ordinal = -1;
  return CudaStatus::Ok;
}

// ---------------------------------------------------------------------------
CudaStatus CudaRunner::UploadVectors(const FPC* host_data,
                                     const IUD* host_ids,
                                     IUC count, IUC dim,
                                     IUC* vram_used_after_mib) {
  if (!inited)        return CudaStatus::NotInit;
  if (!host_data)     return CudaStatus::BadPtr;
  if (!host_ids)      return CudaStatus::BadPtr;
  if (count == 0)     return CudaStatus::Empty;
  if (dim == 0)       return CudaStatus::BadDim;

  const size_t data_bytes = (size_t)count * dim * sizeof(float);
  const size_t ids_bytes  = (size_t)count * sizeof(unsigned long long);
  const size_t score_bytes= (size_t)count * sizeof(double);
  const size_t meta_bytes = (size_t)count * (sizeof(unsigned long long) +
                                             sizeof(unsigned int));
  const size_t query_bytes= (size_t)dim * sizeof(float);

  cudaError_t e;
  if (d_data)    cudaFree(d_data);
  if (d_ids)     cudaFree(d_ids);
  if (d_scores)  cudaFree(d_scores);
  if (d_meta)    cudaFree(d_meta);
  if (d_query)   cudaFree(d_query);
  d_data = d_ids = d_scores = d_meta = d_query = nullptr;

  e = cudaMalloc(&d_data, data_bytes);
  if (e != cudaSuccess) return CudaStatus::AllocFailed;
  e = cudaMalloc(&d_ids, ids_bytes);
  if (e != cudaSuccess) return CudaStatus::AllocFailed;
  e = cudaMalloc(&d_scores, score_bytes);
  if (e != cudaSuccess) return CudaStatus::AllocFailed;
  e = cudaMalloc(&d_meta, meta_bytes);
  if (e != cudaSuccess) return CudaStatus::AllocFailed;
  e = cudaMalloc(&d_query, query_bytes);
  if (e != cudaSuccess) return CudaStatus::AllocFailed;

  // d_meta is (ids_out, slots_out) concatenated: ids first, then slots.
  e = cudaMemcpy(d_data, host_data, data_bytes, cudaMemcpyHostToDevice);
  if (e != cudaSuccess) return MapCudaError(e);
  e = cudaMemcpy(d_ids, host_ids, ids_bytes, cudaMemcpyHostToDevice);
  if (e != cudaSuccess) return MapCudaError(e);

  this->count = count;
  this->dim = dim;

  if (vram_used_after_mib) {
    *vram_used_after_mib =
        (IUC)((data_bytes + ids_bytes + score_bytes + meta_bytes + query_bytes)
              / (1024 * 1024));
  }
  return CudaStatus::Ok;
}

// ---------------------------------------------------------------------------
CudaStatus CudaRunner::SearchTopK(const FPC* host_query, IUC k,
                                  TVectorHit* out, VectorMetric metric) const {
  if (!inited)     return CudaStatus::NotInit;
  if (!host_query) return CudaStatus::BadPtr;
  if (!out)        return CudaStatus::BadPtr;
  if (count == 0)  return CudaStatus::Empty;
  if (k > count)   return CudaStatus::KTooBig;
  if (k == 0)      return CudaStatus::Ok;
  if (k > 1024)    return CudaStatus::KTooBig;  // shared-mem heap bound

  const unsigned int metric_u =
      (metric == VectorMetric::L2Squared) ? 0u : 1u;
  cudaError_t e;

  // Upload the query.
  e = cudaMemcpy(d_query, host_query, (size_t)dim * sizeof(float),
                 cudaMemcpyHostToDevice);
  if (e != cudaSuccess) return MapCudaError(e);

  const IUC blocks = GridForCount(count);
  const IUC threads = kThreadsPerBlock;

  // d_meta layout: [ids_out (count IUD)] [slots_out (count IUC)].
  unsigned long long* d_ids_out = (unsigned long long*)d_meta;
  unsigned int*       d_slots_out =
      (unsigned int*)((char*)d_meta + (size_t)count * sizeof(unsigned long long));

  // Phase 1: score every vector.
  KScore<<<blocks, threads>>>(
      (const float*)d_query, (const float*)d_data,
      (const unsigned long long*)d_ids,
      (double*)d_scores, d_ids_out, d_slots_out, dim, count, metric_u);
  e = cudaGetLastError();
  if (e != cudaSuccess) return MapCudaError(e);

  // Stage 2a: per-block K-best. Needs a device buffer of blocks*K TMeta.
  // Reuse d_scores' sibling allocation: allocate a temporary device buffer
  // for block_best (blocks * K * sizeof(TMeta)).
  const size_t block_best_bytes = (size_t)blocks * k * sizeof(TMeta);
  void* d_block_best = nullptr;
  e = cudaMalloc(&d_block_best, block_best_bytes);
  if (e != cudaSuccess) return CudaStatus::AllocFailed;

  KBlockTopK<<<blocks, threads>>>((const double*)d_scores, d_ids_out,
                                  d_slots_out, (TMeta*)d_block_best,
                                  dim, count, k);
  e = cudaGetLastError();
  if (e != cudaSuccess) { cudaFree(d_block_best); return MapCudaError(e); }

  // Stage 2b: merge the per-block K-best into the global K-best.
  void* d_out_meta = nullptr;
  e = cudaMalloc(&d_out_meta, (size_t)k * sizeof(TMeta));
  if (e != cudaSuccess) { cudaFree(d_block_best); return CudaStatus::AllocFailed; }

  KMergeTopK<<<1, threads>>>((const TMeta*)d_block_best, (TMeta*)d_out_meta,
                             blocks, k);
  e = cudaGetLastError();
  if (e != cudaSuccess) {
    cudaFree(d_block_best); cudaFree(d_out_meta);
    return MapCudaError(e);
  }
  e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    cudaFree(d_block_best); cudaFree(d_out_meta);
    return MapCudaError(e);
  }

  // Copy the K results back.
  TMeta host_meta[1024];
  e = cudaMemcpy(host_meta, d_out_meta, (size_t)k * sizeof(TMeta),
                 cudaMemcpyDeviceToHost);
  cudaFree(d_block_best);
  cudaFree(d_out_meta);
  if (e != cudaSuccess) return MapCudaError(e);

  for (IUC i = 0; i < k; ++i) {
    out[i].score = host_meta[i].score;
    out[i].id = host_meta[i].id;
    out[i].slot = host_meta[i].slot;
  }
  return CudaStatus::Ok;
}

}  // namespace _
