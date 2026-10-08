// Copyright AStarship <https://astarship.net>.
//
// SubsecondDb GPU Vector Database — facade.
//
// The one object the Postgres extension (Pg/) and the iGeek training loop
// talk to. It wires together:
//   * GpuScheduler  — placement/registry (which accelerator, VRAM budget).
//   * VectorIndex   — the exact flat index + CPU reference search.
//   * CudaRunner    — the bare-metal CUDA engine (compiled only when
//                     SUBSECONDBD_HAS_CUDA is defined; absent on CPU boxes).
//   * CrabsTK Data  — TRecord/TReader/TWriter for loading/saving the iGeek
//                     RL-gym training corpora (the "use as much CrabsTK as
//                     possible" hook).
//
// MODE is a runtime choice (NOT a hardware lock), per the design constraints:
//   * VectorDbMode::Cpu  — RAM-resident index, CPU reference search. Runs on
//     any node with or without a GPU.
//   * VectorDbMode::Gpu  — index loaded into an accelerator's VRAM; search
//     runs the CUDA kernels. Falls back to CPU if the runner can't Init.
//
// No C++ std library. ASCII Crabs types only. No exceptions — status codes.

#pragma once
#ifndef SUBSECONDBD_GPU_VECTOR_DB_DECL
#define SUBSECONDBD_GPU_VECTOR_DB_DECL

#include <_Config.h>
#include "VectorIndex.h"
#include "GpuScheduler.h"

// CrabsTK Data API (Record codec + Dataset I/O) for training corpora.
// The include path is set by the build (CrabsTK root on -I). Guarded so a
// build that hasn't wired CrabsTK yet still compiles the core.
//
// The Data API bodies are gated by `#if SEAM >= CRABSTK_DATA_RECORD (52)`.
// The Gpu core builds at SEAM = CRABS_RELEASE (23), which is BELOW that, so
// when the corpus path is on we must raise SEAM to the Data seam level first.
// We do it by redefining SEAM to CRABSTK_DATA_ASCIIPLOT (53) if the current
// value is lower — the Data headers then compile.
#if defined(SUBSECONDBD_USE_CRABSTK_DATA)
#include <Data/_Seams.h>  // CRABSTK_DATA_RECORD (52) / CRABSTK_DATA_ASCIIPLOT (53)
#if SEAM < CRABSTK_DATA_ASCIIPLOT
#undef SEAM
#define SEAM CRABSTK_DATA_ASCIIPLOT
#endif
#include <Data/Record.h>
#include <Data/Dataset.h>
// Implementations: .hxx = non-template function bodies, .hpp = template defs.
#include <Data/Record.hxx>
#include <Data/Dataset.hxx>
#include <Data/Record.hpp>
#include <Data/Dataset.hpp>
#endif

// The CUDA runner is optional at link time. When the CUDA target is built
// (nvcc present) SUBSECONDBD_HAS_CUDA is defined and CudaRunner's methods are
// linked from CudaRunner.cu. On a CPU-only box this is undefined and every
// GPU path returns SchedStatus::NoRunner / VectorDbStatus::NoRunner cleanly.
#ifdef SUBSECONDBD_HAS_CUDA
#include "CudaRunner.h"
#endif

namespace _ {

// ---------------------------------------------------------------------------
// Facade status.
// ---------------------------------------------------------------------------
enum class VectorDbStatus : ISA {
  Ok        =  0,
  NotInit   = -1,
  NoRunner  = -2,  // GPU mode requested but no CUDA runner (CPU box / no GPU)
  Capacity  = -3,  // index full
  BadPtr    = -4,
  BadDim    = -5,
  Empty     = -6,
  KTooBig   = -7,
  Io        = -8,  // CrabsTK corpus I/O failure
  SchedFail = -9,  // scheduler rejected placement (mem/caps/runner)
};

// ---------------------------------------------------------------------------
// VectorDb — the user-facing object.
//
// Capacity/Dim are compile-time bounds on the index (template, so the flat
// buffer is fixed and the CPU/GPU layouts are byte-identical). A typical
// iGeek embedding index: Capacity = 1<<20 (1M vectors), Dim = 384 or 768.
//
// Lifecycle:
//   VectorDb<Capacity, Dim> db;
//   db.Init(VectorMetric::L2Squared, VectorDbMode::Gpu);  // or Cpu
//   db.AddVector(subsecond_id, row);                      // append, FIFO
//   db.SearchTopK(query, k, hits);                        // CPU or GPU
//   db.SaveCorpus(path);  db.LoadCorpus(path);            // CrabsTK TRecord
// ---------------------------------------------------------------------------
template<IUC Capacity, IUC Dim>
struct VectorDb {
  VectorIndex<Capacity, Dim> index;
  GpuScheduler               scheduler;
  VectorMetric               metric = VectorMetric::L2Squared;
  VectorDbMode               mode = VectorDbMode::Cpu;
  BOL                        inited = false;

#ifdef SUBSECONDBD_HAS_CUDA
  CudaRunner runner;
  BOL        runner_up = false;   // GPU path live (Init succeeded)
#endif

  // ---- Init ---------------------------------------------------------------
  // Sets the metric + mode, inits the index, and (GPU mode) brings up the
  // CUDA runner + scheduler registry. GPU mode that can't Init the runner
  // DOWNGRADES to CPU mode and returns NoRunner so the caller knows the
  // search will be CPU (it still works). @return Ok (CPU mode or GPU up),
  // or NoRunner (GPU requested, downgraded to CPU).
  VectorDbStatus Init(VectorMetric m, VectorDbMode desired_mode) {
    metric = m;
    if (index.Init(m) != IndexStatus::Ok) return VectorDbStatus::NotInit;
    scheduler.Init();  // 2-box x 2-GPU registry, Cuda runner active

#ifdef SUBSECONDBD_HAS_CUDA
    runner_up = false;
    if (desired_mode == VectorDbMode::Gpu) {
      CudaStatus rs = runner.Init(0);
      if (rs == CudaStatus::Ok) {
        mode = VectorDbMode::Gpu;
        runner_up = true;
        inited = true;
        return VectorDbStatus::Ok;
      }
      // No GPU / no driver: downgrade to CPU. The search still works.
      mode = VectorDbMode::Cpu;
      inited = true;
      return VectorDbStatus::NoRunner;
    }
    mode = VectorDbMode::Cpu;
    inited = true;
    return VectorDbStatus::Ok;
#else
    // No CUDA at all: any GPU request downgrades to CPU.
    if (desired_mode == VectorDbMode::Gpu) {
      mode = VectorDbMode::Cpu;
      inited = true;
      return VectorDbStatus::NoRunner;
    }
    mode = VectorDbMode::Cpu;
    inited = true;
    return VectorDbStatus::Ok;
#endif
  }

  // ---- Add ----------------------------------------------------------------
  // Append one vector keyed by its SubsecondId. CPU-mode: straight into the
  // index. GPU-mode: into the index AND (lazy) the device is refreshed on the
  // next SearchTopK via a re-upload (batched cold-boot, see SearchTopK).
  VectorDbStatus AddVector(IUD subsecond_id, const FPC* row) {
    if (!inited) return VectorDbStatus::NotInit;
    IndexStatus is = index.AddVector(subsecond_id, row);
    if (is == IndexStatus::Ok)      return VectorDbStatus::Ok;
    if (is == IndexStatus::Capacity) return VectorDbStatus::Capacity;
    if (is == IndexStatus::NotInit)  return VectorDbStatus::NotInit;
    return VectorDbStatus::BadPtr;
  }

  // ---- Search -------------------------------------------------------------
  // Top-k search. GPU-mode with a live runner uploads any new vectors
  // (cold-boot if the device count < index count) then runs the kernels;
  // otherwise the CPU reference. Either way out[0..k-1] is ascending by
  // score (best first) and is the SAME ranking the CPU reference produces.
  VectorDbStatus SearchTopK(const FPC* query, IUC k, TVectorHit* out) {
    if (!inited)  return VectorDbStatus::NotInit;
    if (!query)   return VectorDbStatus::BadPtr;
    if (!out)     return VectorDbStatus::BadPtr;
    if (index.Empty()) return VectorDbStatus::Empty;
    if (k > index.Count()) return VectorDbStatus::KTooBig;

#ifdef SUBSECONDBD_HAS_CUDA
    if (mode == VectorDbMode::Gpu && runner_up) {
      // Cold-boot / refresh: if the device has fewer vectors than the index,
      // re-upload the whole index (the index is append-only and the buffer is
      // row-major, so a full re-upload is correct; a delta-upload is a later
      // optimization). This is the RAM->VRAM transfer the scheduler budgets.
      if (runner.count < index.Count()) {
        IUC vram_after = 0;
        CudaStatus us = runner.UploadVectors(index.data_, index.ids_,
                                             index.Count(), Dim, &vram_after);
        if (us != CudaStatus::Ok) {
          // Upload failed (e.g. out of VRAM): fall back to CPU for this query
          // so the search still answers. (A production path would return
          // SchedFail; the fallback keeps iGeek training from stalling.)
          mode = VectorDbMode::Cpu;
          runner_up = false;
        }
      }
      if (mode == VectorDbMode::Gpu && runner_up) {
        CudaStatus rs = runner.SearchTopK(query, k, out, metric);
        if (rs == CudaStatus::Ok) return VectorDbStatus::Ok;
        // Kernel error: fall back to CPU so the answer is still correct.
        mode = VectorDbMode::Cpu;
        runner_up = false;
      }
    }
#endif
    // CPU reference (always correct; also the GPU fallback).
    IndexStatus is = index.SearchTopK(query, k, out);
    if (is == IndexStatus::Ok)        return VectorDbStatus::Ok;
    if (is == IndexStatus::Empty)     return VectorDbStatus::Empty;
    if (is == IndexStatus::KTooBig)   return VectorDbStatus::KTooBig;
    if (is == IndexStatus::BadPtr)    return VectorDbStatus::BadPtr;
    return VectorDbStatus::NotInit;
  }

  // ---- Queries ------------------------------------------------------------
  IUC  Count() const { return index.Count(); }
  BOL  Empty() const { return index.Empty(); }
  VectorDbMode Mode() const { return mode; }

  // Cold-boot budget (ms) the scheduler predicts for loading the current
  // index size into VRAM. CPU mode -> 0.
  ISD ColdBootBudgetMs() const {
    if (mode != VectorDbMode::Gpu) return 0;
    IUC bytes_mib = (IUC)((index.Count() * Dim * (sizeof(FPC) +
                                                   sizeof(IUD))) /
                          (1024 * 1024));
    return GpuScheduler::ColdBootBudgetMs(bytes_mib);
  }

  // ---- CrabsTK corpus I/O -------------------------------------------------
  //
  // The iGeek RL-gym training loop produces TRecord batches (obs/action/
  // reward/terminal rows). VectorDb.LoadCorpus/SaveCorpus read/write those
  // batches to/from disk via the CrabsTK Data API, so the training pipeline
  // and the vector DB share one on-disk format. The vectors live in the
  // TRecord's obs[] field (ObsLength == Dim); each row's SubsecondId is the
  // run_id/sequence the RL loop assigned.
  //
  // Requires SUBSECONDBD_USE_CRABSTK_DATA. On a build without it these
  // return Io (the corpus path is unavailable, not an error state).
#ifdef SUBSECONDBD_USE_CRABSTK_DATA
  // Load a .crbroll corpus file into the index (appends to existing).
  // Capacity/ObsLength/ActionCount/StateWords must match the corpus.
  template<IUC ObsLength, IUC ActionCount, IUC StateWords = 0>
  VectorDbStatus LoadCorpus(const CHA* path) {
    if (!inited) return VectorDbStatus::NotInit;
    static_assert(ObsLength == Dim,
                  "corpus ObsLength must equal the index Dim");
    TReader reader;
    RecordStatus rs = TReaderOpen(reader, path);
    if (rs != RecordOK) { TReaderClose(reader); return VectorDbStatus::Io; }

    // Bounded scratch for ONE record of this shape. Sized to the checked wire
    // size (header + Capacity rows), NOT RecordMaxBytes (16 MB) — a 16 MB
    // stack frame segfaults. The formula matches TRecord's static_assert and
    // is a compile-time constant from the template params (no VLA).
    TRecord<Capacity, ObsLength, ActionCount, StateWords> rec;
    constexpr IUC kScratchBytes =
        RecordHeaderBytes +
        Capacity * (37 + ObsLength * 4 + ActionCount +
                    (StateWords ? StateWords : 1) * 4);
    IUA scratch[kScratchBytes];
    IUD loaded = 0;
    for (;;) {
      // TReaderRead advances the reader (it calls TReaderNext internally) AND
      // decodes into rec. Drive the loop with it — calling TReaderNext first
      // would double-advance and skip every other record. RecordEnd on clean
      // EOF terminates the loop.
      rs = TReaderRead(reader, rec, scratch, kScratchBytes);
      if (rs == RecordEnd) break;          // clean EOF
      if (rs != RecordOK) { TReaderClose(reader); return VectorDbStatus::Io; }
      for (IUC i = 0; i < rec.count; ++i) {
        // Per-row SubsecondId was stored in episode_ids[i] by SaveCorpus.
        IUD sid = rec.episode_ids[i];
        if (index.AddVector(sid, rec.obs + i * ObsLength)
            != IndexStatus::Ok) {
          // Index full: stop loading (the corpus is bigger than Capacity).
          TReaderClose(reader);
          return VectorDbStatus::Capacity;
        }
        ++loaded;
      }
    }
    TReaderClose(reader);
    (void)loaded;
    return VectorDbStatus::Ok;
  }

  // Save the current index as a single-record .crbroll corpus.
  template<IUC ActionCount = 1, IUC StateWords = 0>
  VectorDbStatus SaveCorpus(const CHA* path) const {
    if (!inited)        return VectorDbStatus::NotInit;
    if (index.Empty())  return VectorDbStatus::Empty;
    TWriter writer;
    RecordStatus rs = TWriterOpen(writer, path);
    if (rs != RecordOK) { TWriterClose(writer); return VectorDbStatus::Io; }

    TRecord<Capacity, Dim, ActionCount, StateWords> rec;
    rec.run_id = 1;
    rec.sequence = 0;
    rec.seed = 0;
    rec.count = index.Count();
    const IUD* row_ids = index.Ids();
    const FPC* row_data = index.Data();
    for (IUC i = 0; i < index.Count(); ++i) {
      // The per-row SubsecondId lives in episode_ids[i] (the only per-row IUD
      // field TRecord has) — TRecord's run_id/sequence are per-batch, not
      // per-row, so they can't carry a per-vector id.
      rec.episode_ids[i] = row_ids[i];
      rec.env_ids[i] = 0;
      rec.steps[i] = 0;
      for (IUC d = 0; d < Dim; ++d) rec.obs[i * Dim + d] = row_data[i * Dim + d];
      rec.actions[i] = 0;
      rec.rewards[i] = 0.0f;
      rec.terminals[i] = 0;
      for (IUC a = 0; a < ActionCount; ++a)
        rec.action_mask[i * ActionCount + a] = 1;
      rec.values[i] = 0.0f;
      rec.logp[i] = 0.0f;
      for (IUC s = 0; s < (StateWords ? StateWords : 1); ++s)
        rec.states[i * (StateWords ? StateWords : 1) + s] = 0;
    }
    // Same compile-time-sized scratch as LoadCorpus (one record of this shape).
    constexpr IUC kScratchBytes =
        RecordHeaderBytes +
        Capacity * (37 + Dim * 4 + ActionCount +
                    (StateWords ? StateWords : 1) * 4);
    IUA scratch[kScratchBytes];
    IUC written = 0;
    rs = TWriterAppend(writer, rec, scratch, kScratchBytes);
    if (rs != RecordOK) { TWriterClose(writer); return VectorDbStatus::Io; }
    rs = TWriterClose(writer);
    (void)written;
    return rs == RecordOK ? VectorDbStatus::Ok : VectorDbStatus::Io;
  }
#else
  VectorDbStatus LoadCorpus(const CHA*) { return VectorDbStatus::Io; }
  VectorDbStatus SaveCorpus(const CHA*) const { return VectorDbStatus::Io; }
#endif
};

}  // namespace _

#endif  // SUBSECONDBD_GPU_VECTOR_DB_DECL
