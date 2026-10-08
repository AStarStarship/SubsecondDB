// Copyright AStarship <https://astarship.net>.
//
// SubsecondDb GPU Vector Database — flat exact ANN index.
//
// What this is: the data layout + exact top-k search for a vector index.
// Vectors are FPC (float32) rows keyed by IUD SubsecondId (64-bit,
// time-ordered, so the index is append-only and FIFO-consistent by
// construction). Search is EXACT top-k (brute force over the index) — a
// correct baseline. IVF / HNSW / product-quantization are later optimizations
// and are NOT faked here.
//
// Layout: vectors are stored row-major (vector i is dims[i*Dim .. i*Dim+Dim-1])
// so a query scans a contiguous FPC array — the same layout the CUDA runner
// consumes, so CPU and GPU paths are byte-identical.
//
// No C++ std library. ASCII Crabs types only. No exceptions — status codes.
//
// WHAT THIS MODULE IS NOT:
//   - Not the CUDA kernel (that is CudaRunner).
//   - Not the scheduler (that is GpuScheduler). This is the index + the CPU
//     reference search the GPU result is cross-checked against.

#pragma once
#ifndef SUBSECONDBD_GPU_VECTOR_INDEX_DECL
#define SUBSECONDBD_GPU_VECTOR_INDEX_DECL

#include <_Config.h>

namespace _ {

// ---------------------------------------------------------------------------
// Distance metric.
// ---------------------------------------------------------------------------
enum class VectorMetric : IUB {
  L2Squared  = 0,  // Sum of squared differences. No sqrt — cheaper, and for a
                   // fixed query the argmin over L2sq == argmin over L2.
  InnerProduct = 1,  // Negative dot product, so "top-k nearest" = largest dot.
};

// ---------------------------------------------------------------------------
// Index status — no exceptions, return codes.
// ---------------------------------------------------------------------------
enum class IndexStatus : ISA {
  Ok         =  0,
  Empty      = -1,  // Index has no vectors yet
  BadDim     = -2,  // Dim mismatch between query and index
  Capacity   = -3,  // Index is full (bounded flat array)
  BadPtr     = -4,  // Null pointer
  NotInit    = -5,  // Index not initialized
  KTooBig    = -6,  // k > number of vectors
};

// ---------------------------------------------------------------------------
// A search result row: the SubsecondId + its score. "score" is the distance
// (lower = nearer) for L2Squared, or the negative inner product (lower =
// nearer, i.e. larger dot) for InnerProduct — so results are ALWAYS
// ascending by score = "most relevant first".
// ---------------------------------------------------------------------------
struct TVectorHit {
  IUD   id;       // SubsecondId of the matched vector
  FPD   score;    // distance (L2sq) or negative dot (IP); ascending = better
  IUC   slot;     // internal slot index (for debugging / verification)
};

// ---------------------------------------------------------------------------
// VectorIndex — bounded, flat, append-only exact index.
//
// CAPACITY is a compile-time bound on the number of vectors. The index is a
// fixed FPC buffer + IUD id buffer; AddVector appends until full. This keeps
// the layout POD-friendly and identical for the CPU reference and the CUDA
// runner (which memcpys the same buffer to VRAM on cold-boot).
// ---------------------------------------------------------------------------
template<IUC Capacity, IUC Dim>
struct VectorIndex {
  // Capacity/Dim sanity — enforced at the use site via static_assert in the
  // methods (keeps this header light).
  FPC  data_[Capacity * Dim];  // row-major vectors
  IUD  ids_[Capacity];         // SubsecondId per slot
  IUC  count = 0;              // vectors currently stored
  BOL  inited = false;

  // ---- Init ---------------------------------------------------------------
  // Zero the index. Capacity/Dim are template params; this just resets state.
  IndexStatus Init(VectorMetric metric) {
    static_assert(Capacity > 0 && Dim > 0, "Capacity and Dim must be > 0");
    for (IUC i = 0; i < Capacity * Dim; ++i) data_[i] = 0.0f;
    for (IUC i = 0; i < Capacity; ++i) ids_[i] = 0;
    count = 0;
    metric_ = metric;
    inited = true;
    return IndexStatus::Ok;
  }

  // ---- Add ----------------------------------------------------------------
  // Append one vector keyed by its SubsecondId. The caller owns the FPC row
  // (Dim floats). Returns Capacity when full. SubsecondId is the primary key;
  // because it is time-ordered, append order == id order == FIFO.
  IndexStatus AddVector(IUD subsecond_id, const FPC* row) {
    if (!inited)            return IndexStatus::NotInit;
    if (!row)               return IndexStatus::BadPtr;
    if (count >= Capacity)  return IndexStatus::Capacity;
    for (IUC d = 0; d < Dim; ++d) data_[count * Dim + d] = row[d];
    ids_[count] = subsecond_id;
    ++count;
    return IndexStatus::Ok;
  }

  // ---- Queries ------------------------------------------------------------
  IUC  Count() const { return count; }
  // Per-row SubsecondId in append order (for corpus Save). Null when empty.
  const IUD* Ids() const { return ids_; }
  // Per-row vector data pointer (row i = data_ + i*Dim), for corpus Save.
  const FPC* Data() const { return data_; }
  BOL  Empty() const { return count == 0; }
  IUC  DimOf() const { return Dim; }

  // ---- Exact top-k search (CPU reference) ---------------------------------
  //
  // Scans every stored vector, computes its score against the query, and
  // keeps the k lowest scores using a bounded max-heap of size k (no
  // std::priority_queue — a hand-rolled binary heap over a fixed array).
  // Results are written ascending by score (best first) into out[0..k-1].
  //
  // @pre inited, count > 0, k <= count.
  // @return Ok, or Empty / KTooBig / BadPtr.
  IndexStatus SearchTopK(const FPC* query, IUC k, TVectorHit* out) const {
    if (!inited)      return IndexStatus::NotInit;
    if (!query)       return IndexStatus::BadPtr;
    if (!out)         return IndexStatus::BadPtr;
    if (count == 0)   return IndexStatus::Empty;
    if (k > count)    return IndexStatus::KTooBig;
    if (k == 0)       return IndexStatus::Ok;  // nothing to do

    // Bounded max-heap of the k best (lowest) scores seen so far. The heap
    // root is the WORST (highest) of the current k; anything worse than the
    // root is discarded, anything better replaces the root.
    TVectorHit heap[k > Capacity ? Capacity : k];
    IUC heap_n = 0;
    const IUC kcap = k > Capacity ? Capacity : k;

    for (IUC i = 0; i < count; ++i) {
      FPD score = Score(query, data_ + i * Dim, Dim, metric_);
      TVectorHit cand;
      cand.id = ids_[i];
      cand.score = score;
      cand.slot = i;
      if (heap_n < kcap) {
        heap[heap_n] = cand;
        ++heap_n;
        SiftUp(heap, heap_n, heap_n - 1);
      } else if (score < heap[0].score) {
        heap[0] = cand;
        SiftDown(heap, kcap, 0);
      }
    }

    // Drain the max-heap. Popping a max-heap yields the WORST (highest score)
    // first, i.e. descending. We want ascending (best first). The heap lives in
    // tmp[]; popped (descending) results go into a SEPARATE desc[] array so the
    // in-place pop (which overwrites tmp[0]) never clobbers a recorded value.
    const IUC total = kcap;
    TVectorHit tmp[kcap];
    for (IUC i = 0; i < total; ++i) tmp[i] = heap[i];
    TVectorHit desc[kcap];
    IUC n = total;
    for (IUC i = 0; i < total; ++i) {
      desc[i] = tmp[0];                  // current root = max (worst) so far
      TVectorHit last = tmp[n - 1];      // last live element
      --n;
      if (n > 0) {
        tmp[0] = last;                   // promote last to root
        SiftDown(tmp, n, 0);
      }
    }
    // desc[] is descending (worst at [0], best at [total-1]). Reverse into out.
    for (IUC i = 0; i < total; ++i) out[i] = desc[total - 1 - i];
    return IndexStatus::Ok;
  }

  // ---- Score (CPU reference) ---------------------------------------------
  // The EXACT metric the CUDA kernel must reproduce, selected by the index's
  // metric. L2Squared: sum of squared diffs (lower = nearer). InnerProduct:
  // negative dot (lower = nearer, i.e. larger dot). Computed in FPD (double)
  // to match the CUDA kernel's accumulation; the argmin is metric-identical.
  static FPD Score(const FPC* a, const FPC* b, IUC dim, VectorMetric m) {
    FPD acc = 0.0;
    if (m == VectorMetric::L2Squared) {
      for (IUC d = 0; d < dim; ++d) {
        FPD diff = (FPD)a[d] - (FPD)b[d];
        acc += diff * diff;
      }
    } else {  // InnerProduct: negative dot
      for (IUC d = 0; d < dim; ++d) {
        acc += (FPD)a[d] * (FPD)b[d];
      }
      acc = -acc;
    }
    return acc;
  }

  VectorMetric metric_ = VectorMetric::L2Squared;

 private:
  // Max-heap sift-up (parent = (i-1)/2).
  static void SiftUp(TVectorHit* h, IUC n, IUC i) {
    while (i > 0) {
      IUC p = (i - 1) / 2;
      if (h[p].score >= h[i].score) break;
      TVectorHit t = h[p]; h[p] = h[i]; h[i] = t;
      i = p;
    }
  }
  // Max-heap sift-down.
  static void SiftDown(TVectorHit* h, IUC n, IUC i) {
    for (;;) {
      IUC l = 2 * i + 1, r = 2 * i + 2, m = i;
      if (l < n && h[l].score > h[m].score) m = l;
      if (r < n && h[r].score > h[m].score) m = r;
      if (m == i) break;
      TVectorHit t = h[m]; h[m] = h[i]; h[i] = t;
      i = m;
    }
  }
};

}  // namespace _

#endif  // SUBSECONDBD_GPU_VECTOR_INDEX_DECL
