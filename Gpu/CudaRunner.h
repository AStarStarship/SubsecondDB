// Copyright AStarship <https://astarship.net>.
//
// SubsecondDb GPU Vector Database — bare-metal CUDA runner.
//
// This is the RUNNER the scheduler routes vector-search jobs to. It owns the
// GPU memory and drives the actual top-k kernels. It is the ONLY place in the
// module that touches <cuda_runtime.h> — the scheduler, index, and facade are
// CUDA-free, so the whole CPU path builds and tests on a box with no GPU.
//
// BARE METAL: <cuda_runtime.h> only. No CUDA toolkit libs, no thrust, no
// cub, no cuBLAS. Distance + top-k are hand-rolled __global__ kernels that
// reproduce VectorIndex::Score exactly (L2Squared = sum of squared diffs,
// InnerProduct = negative dot), so the GPU result is cross-checkable against
// the CPU reference bit-for-bit in the ranking (scores match within FPC
// accumulation tolerance).
//
// THREAD MODEL (one search):
//   * Phase 1 — every block handles a slice of the index; each thread scores
//     one vector and writes (score, id, slot) to a global scratch buffer.
//   * Phase 2 — a single-block top-k selection over the scratch buffer using
//     a shared-memory bounded max-heap (size K), mirroring the CPU heap so
//     the ranking is identical.
//
// All host APIs return CudaStatus (no exceptions). The runner is single
// context per accelerator; the scheduler's AccelState/VRAM accounting tracks
// the same memory this runner allocates.

#pragma once
#ifndef SUBSECONDBD_GPU_CUDA_RUNNER_DECL
#define SUBSECONDBD_GPU_CUDA_RUNNER_DECL

#include <_Config.h>
#include "VectorIndex.h"  // TVectorHit, VectorMetric, FPC/IUD types

namespace _ {

// ---------------------------------------------------------------------------
// Runner status — no exceptions.
// ---------------------------------------------------------------------------
enum class CudaStatus : ISA {
  Ok          =  0,
  NoDevice    = -1,  // cudaGetDeviceCount returned 0 / no GPU
  AllocFailed = -2,  // cudaMalloc / copy failed
  BadPtr      = -3,  // null argument
  BadDim      = -4,  // dim mismatch
  NotInit     = -5,  // runner not initialized
  KTooBig     = -6,  // k > count
  Empty       = -7,  // no vectors
  CudaError   = -8,  // generic cudaError_t != success
};

// ---------------------------------------------------------------------------
// CudaRunner — one accelerator's vector-search engine.
//
// Init() probes the device and reports VRAM/device name (populated into the
// GpuScheduler registry by the facade). UploadVectors() copies the index's
// FPC buffer + IUD id buffer to VRAM (this IS the cold-boot the scheduler
// budgets for). SearchTopK() runs the two-phase kernel and returns hits in
// the SAME ascending-score order as the CPU reference.
//
// The runner is deliberately POD-state (device pointers + sizes) so the
// facade can hand it to the scheduler for VRAM accounting without any
// CUDA-specific type leaking across the scheduler boundary.
// ---------------------------------------------------------------------------
struct CudaRunner {
  // Device handles (opaque; 0/null == unallocated).
  void*  d_data = nullptr;   // device FPC buffer (count*Dim floats)
  void*  d_ids  = nullptr;   // device IUD buffer (count SubsecondIds)
  void*  d_scores = nullptr; // device FPD scratch (count) — phase-1 output
  void*  d_meta = nullptr;   // device (IUD id, IUC slot) scratch (count)
  void*  d_query = nullptr;  // device FPC buffer (Dim floats)

  IUC    count = 0;          // vectors currently on-device
  IUC    dim = 0;
  ISN    device_ordinal = -1;
  BOL    inited = false;

  // Device description (filled by Init) — the facade copies these into the
  // GpuScheduler registry (name, vram_total_mib).
  CHA    device_name_[64];
  IUC    vram_total_mib = 0;
  IUC    vram_free_mib = 0;
  IUB    compute_capability_major = 0;
  IUB    compute_capability_minor = 0;

  // ---- Lifecycle ----------------------------------------------------------
  // Probe the device. ordinal < 0 = auto-pick device 0. Populates
  // device_name_/vram_* / compute_capability_*. No allocation here.
  CudaStatus Init(ISN ordinal = 0);

  // Release all device memory. Idempotent.
  CudaStatus Shutdown();

  // ---- Upload (cold-boot) -------------------------------------------------
  // Copy count*Dim FPC + count IUD to VRAM, growing/reallocating device
  // buffers as needed. This is the RAM->VRAM transfer the scheduler budgets.
  // vram_used_after_mib is set to the bytes allocated / 1MiB on success.
  CudaStatus UploadVectors(const FPC* host_data, const IUD* host_ids,
                           IUC count, IUC dim, IUC* vram_used_after_mib);

  // ---- Search -------------------------------------------------------------
  // Run the two-phase top-k on-device. out[0..k-1] is ascending by score,
  // identical ranking to VectorIndex::SearchTopK (scores match within FPC
  // accumulation tolerance). @pre inited, count>0, k<=count, dim matches.
  CudaStatus SearchTopK(const FPC* host_query, IUC k, TVectorHit* out,
                        VectorMetric metric) const;
};

}  // namespace _

#endif  // SUBSECONDBD_GPU_CUDA_RUNNER_DECL
