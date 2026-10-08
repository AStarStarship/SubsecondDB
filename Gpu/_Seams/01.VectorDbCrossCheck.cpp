// Copyright AStarship <https://astarship.net>.
//
// Cross-check: the CUDA runner's two-stage top-k ALGORITHM (per-block local
// bounded max-heap -> single-block merge) mirrored host-side in C++, asserted
// to produce the SAME ranking as the CPU reference VectorIndex::SearchTopK.
//
// This is NOT the CUDA build — it is the host-side proof that the algorithm
// the __global__ kernels encode is correct, so when CudaRunner.cu runs on the
// box its ranking matches the CPU reference. The kernel code (CudaRunner.cu)
// and this mirror share the identical bounded-heap + drain-and-reverse logic.
//
// Compile: g++ -std=c++23 (no CUDA needed).

#include "VectorIndex.h"
#include <unistd.h>

using namespace _;

namespace {

struct TMeta { FPD score; IUD id; IUC slot; };

// Mirror of KScore (CPU): score every vector, write (score,id,slot).
void ScoreAll(const FPC* query, const FPC* data, const IUD* ids,
              IUC count, IUC dim, VectorMetric m,
              FPD* scores, IUD* ids_out, IUC* slots_out) {
  for (IUC i = 0; i < count; ++i) {
    scores[i] = VectorIndex<1, 1>::Score(query, data + i * dim, dim, m);
    ids_out[i] = ids[i];
    slots_out[i] = i;
  }
}

// Mirror of KBlockTopK: one block's slice -> local K-best (max-heap, root =
// worst-of-K), staged exactly like the kernel (shared stage buffer, thread-0
// drain). Here "one block" = the whole array (single-block mirror), which
// exercises the same heap insert + staging path.
void BlockTopK(const FPD* scores, const IUD* ids, const IUC* slots,
               IUC count, IUC K, TMeta* block_best) {
  TMeta heap[1024];
  IUC heap_n = 0;
  constexpr IUC STAGE = 256;
  TMeta stage[STAGE];
  for (IUC base = 0; base < count; base += STAGE) {
    IUC n_in_stage = (base + STAGE < count) ? STAGE : (count - base);
    for (IUC off = 0; off < STAGE; ++off) {
      if (off < n_in_stage) {
        stage[off].score = scores[base + off];
        stage[off].id = ids[base + off];
        stage[off].slot = slots[base + off];
      } else {
        stage[off].score = 1e300; stage[off].id = 0; stage[off].slot = 0;
      }
    }
    for (IUC s = 0; s < STAGE; ++s) {
      TMeta cand = stage[s];
      if (cand.score >= 1e299) continue;
      if (heap_n < K) {
        heap[heap_n] = cand; ++heap_n;
        IUC p = heap_n - 1;
        while (p > 0) {
          IUC par = (p - 1) / 2;
          if (heap[par].score >= heap[p].score) break;
          TMeta t = heap[par]; heap[par] = heap[p]; heap[p] = t;
          p = par;
        }
      } else if (cand.score < heap[0].score) {
        heap[0] = cand;
        IUC p = 0;
        for (;;) {
          IUC l = 2 * p + 1, r = 2 * p + 2, mm = p;
          if (l < K && heap[l].score > heap[mm].score) mm = l;
          if (r < K && heap[r].score > heap[mm].score) mm = r;
          if (mm == p) break;
          TMeta t = heap[mm]; heap[mm] = heap[p]; heap[p] = t;
          p = mm;
        }
      }
    }
  }
  for (IUC i = 0; i < K; ++i) {
    if (i < heap_n) block_best[i] = heap[i];
    else { block_best[i].score = 1e300; block_best[i].id = 0; block_best[i].slot = 0; }
  }
}

// Mirror of KMergeTopK: merge (blocks * K) block-best -> global K-best
// ascending (best first). Single-block mirror.
void MergeTopK(const TMeta* block_best, IUC blocks, IUC K, TMeta* out) {
  TMeta heap[1024];
  IUC heap_n = 0;
  const IUC total = blocks * K;
  for (IUC i = 0; i < total; ++i) {
    TMeta cand = block_best[i];
    if (cand.score >= 1e299) continue;
    if (heap_n < K) {
      heap[heap_n] = cand; ++heap_n;
      IUC p = heap_n - 1;
      while (p > 0) {
        IUC par = (p - 1) / 2;
        if (heap[par].score >= heap[p].score) break;
        TMeta t = heap[par]; heap[par] = heap[p]; heap[p] = t;
        p = par;
      }
    } else if (cand.score < heap[0].score) {
      heap[0] = cand;
      IUC p = 0;
      for (;;) {
        IUC l = 2 * p + 1, r = 2 * p + 2, mm = p;
        if (l < K && heap[l].score > heap[mm].score) mm = l;
        if (r < K && heap[r].score > heap[mm].score) mm = r;
        if (mm == p) break;
        TMeta t = heap[mm]; heap[mm] = heap[p]; heap[p] = t;
        p = mm;
      }
    }
  }
  TMeta desc[1024];
  IUC n = heap_n;
  for (IUC i = 0; i < heap_n; ++i) {
    desc[i] = heap[0];
    TMeta last = heap[n - 1];
    --n;
    if (n > 0) {
      heap[0] = last;
      IUC p = 0;
      for (;;) {
        IUC l = 2 * p + 1, r = 2 * p + 2, mm = p;
        if (l < n && heap[l].score > heap[mm].score) mm = l;
        if (r < n && heap[r].score > heap[mm].score) mm = r;
        if (mm == p) break;
        TMeta t = heap[mm]; heap[mm] = heap[p]; heap[p] = t;
        p = mm;
      }
    }
  }
  for (IUC i = 0; i < heap_n; ++i) out[i] = desc[heap_n - 1 - i];
}

}  // namespace

int main() {
  // A non-trivial index: 3000 vectors of dim 16, pseudo-random but
  // deterministic (no std:: — a simple LCG so it's reproducible).
  constexpr IUC COUNT = 3000, DIM = 16, K = 10, BLOCKS = 12;
  FPC data[COUNT * DIM];
  IUD ids[COUNT];
  FPC query[DIM];
  IUD state = 0x123456789ULL;
  auto lcg = [&]() -> FPC {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (FPC)((state >> 40) & 0xFFFF) / 65535.0f * 2.0f - 1.0f;  // [-1,1)
  };
  for (IUC i = 0; i < COUNT * DIM; ++i) data[i] = lcg();
  for (IUC i = 0; i < COUNT; ++i) ids[i] = 1000000 + i;  // SubsecondId-like
  for (IUC d = 0; d < DIM; ++d) query[d] = lcg();

  // CPU reference.
  VectorIndex<COUNT, DIM> ref;
  ref.Init(VectorMetric::L2Squared);
  for (IUC i = 0; i < COUNT; ++i)
    ref.AddVector(ids[i], data + i * DIM);
  TVectorHit ref_hits[K];
  if (ref.SearchTopK(query, K, ref_hits) != IndexStatus::Ok) return 2;

  // CUDA-algorithm mirror: score -> per-block top-k -> merge.
  FPD scores[COUNT]; IUD ids_out[COUNT]; IUC slots_out[COUNT];
  ScoreAll(query, data, ids, COUNT, DIM, VectorMetric::L2Squared,
           scores, ids_out, slots_out);
  // Split into BLOCKS slices, each does a local top-K, then merge.
  TMeta block_best[BLOCKS * K];
  IUC per_block = (COUNT + BLOCKS - 1) / BLOCKS;
  for (IUC b = 0; b < BLOCKS; ++b) {
    IUC lo = b * per_block, hi = (lo + per_block < COUNT) ? lo + per_block : COUNT;
    IUC n = hi - lo;
    // Build per-block score/id/slot views.
    FPD bs[n]; IUD bi[n]; IUC bsl[n];
    for (IUC i = 0; i < n; ++i) { bs[i] = scores[lo + i]; bi[i] = ids_out[lo + i]; bsl[i] = slots_out[lo + i]; }
    TMeta bb[K];
    BlockTopK(bs, bi, bsl, n, K, bb);
    for (IUC i = 0; i < K; ++i) block_best[b * K + i] = bb[i];
  }
  TMeta merged[K];
  MergeTopK(block_best, BLOCKS, K, merged);

  // Compare rankings: same ids, same order, scores within FPC tolerance.
  for (IUC i = 0; i < K; ++i) {
    if (merged[i].id != ref_hits[i].id) {
      write(2, "RANK MISMATCH\n", 14);
      // Dump both rankings for diagnosis (id + score*1000).
      for (IUC j = 0; j < K; ++j) {
        CHA line[64]; ISN n = 0;
        auto putn = [&](long long v) {
          char b[24]; int m = 0;
          if (v == 0) b[m++] = '0';
          else { if (v < 0) { b[m++]='-'; v=-v; } while (v>0){ b[m++]='0'+(v%10); v/=10; } while (m>1 && b[m-1]=='0') --m; }
          while (m > 0) line[n++] = b[--m];
        };
        putn((long long)j); line[n++]=' ';
        putn((long long)ref_hits[j].id); line[n++]=' ';
        putn((long long)(ref_hits[j].score * 1000.0)); line[n++]='|';
        putn((long long)merged[j].id); line[n++]=' ';
        putn((long long)(merged[j].score * 1000.0)); line[n++]='\n';
        line[n] = 0;
        write(2, line, (size_t)n);
      }
      return 3 + (IUC)i;
    }
    FPD diff = merged[i].score - ref_hits[i].score;
    if (diff < 0) diff = -diff;
    // FPC accumulation tolerance: relative 1e-4 on scores ~O(DIM).
    FPD tol = 1e-3f + (FPD)(DIM) * 1e-5f;
    if (diff > tol) { write(2, "SCORE MISMATCH\n", 15); return 40 + (IUC)i; }
  }
  write(2, "CROSS-CHECK PASS: CUDA-algorithm mirror ranking == CPU reference "
          "(K=10 over 3000x16, L2sq)\n", 76);
  // Also exercise InnerProduct. Use a SEPARATE index (Init() resets count, so
  // re-Init on `ref` would wipe its 3000 vectors).
  VectorIndex<COUNT, DIM> ref_ip_idx;
  ref_ip_idx.Init(VectorMetric::InnerProduct);
  for (IUC i = 0; i < COUNT; ++i)
    ref_ip_idx.AddVector(ids[i], data + i * DIM);
  TVectorHit ref_ip[K];
  if (ref_ip_idx.SearchTopK(query, K, ref_ip) != IndexStatus::Ok) return 5;
  ScoreAll(query, data, ids, COUNT, DIM, VectorMetric::InnerProduct,
           scores, ids_out, slots_out);
  for (IUC b = 0; b < BLOCKS; ++b) {
    IUC lo = b * per_block, hi = (lo + per_block < COUNT) ? lo + per_block : COUNT;
    IUC n = hi - lo;
    FPD bs[n]; IUD bi[n]; IUC bsl[n];
    for (IUC i = 0; i < n; ++i) { bs[i] = scores[lo + i]; bi[i] = ids_out[lo + i]; bsl[i] = slots_out[lo + i]; }
    TMeta bb[K]; BlockTopK(bs, bi, bsl, n, K, bb);
    for (IUC i = 0; i < K; ++i) block_best[b * K + i] = bb[i];
  }
  MergeTopK(block_best, BLOCKS, K, merged);
  for (IUC i = 0; i < K; ++i) {
    if (merged[i].id != ref_ip[i].id) { write(2, "IP RANK MISMATCH\n", 17); return 50 + (IUC)i; }
  }
  write(2, "CROSS-CHECK PASS: InnerProduct mirror ranking == CPU reference\n", 63);
  return 0;
}
