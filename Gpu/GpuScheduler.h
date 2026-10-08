// Copyright AStarship <https://astarship.net>.
//
// SubsecondDb GPU Scheduler — backend-agnostic core.
//
// HARDWARE (two identical boxes, dev + prod):
//   Xeon Gold 6230, 96 GB RAM @ 130 GB/s, 1x V100 16 GB, 1x RTX 5060 Ti 16 GB.
//
// DESIGN CONSTRAINTS (from the Captain):
//   1. Vector DB can be CPU-only OR GPU-accelerated. This is a runtime mode
//      choice, NOT a hardware lock. A node with no GPU still runs the vector
//      DB in CPU mode (RAM-resident index).
//   2. The GPU RUNNER is CUDA-only *for now*. That is acknowledged and fine,
//      but CUDA must not be STRUCTURAL: the scheduler matches jobs to
//      CAPABILITIES, and the runner backend is a pluggable tag. Swapping the
//      runner to HIP/LevelZero/OpenCL later is a runner replacement, not a
//      scheduler rewrite.
//   3. No C++ std library. ASCII Crabs types only. No exceptions — status codes.
//
// WHAT THIS MODULE IS:
//   The scheduler brain. It knows which accelerator is free, which
//   capabilities a job needs, and whether the memory (VRAM or RAM) fits.
//   It does NOT drive any GPU — that is the runner's job (CUDA today).
//
// WHAT THIS MODULE IS NOT:
//   - Not a CUDA library. It never includes <cuda_runtime.h>.
//   - Not a Postgres extension (yet). Pure C++ core, seam-tested.
//   - Not a DDP/training federator. The "supercomputer" is the Hermes Agent
//     pooling the two boxes; this is the placement/registry brain.

#pragma once
#ifndef SUBSECONDBD_GPU_SCHEDULER_DECL
#define SUBSECONDBD_GPU_SCHEDULER_DECL

#include <_Config.h>

namespace _ {

// ---------------------------------------------------------------------------
// Capability bits — what an accelerator can DO, not what it IS.
// A job requires a subset of these; an accelerator advertises a superset.
// Placement = subset test: (gpu.caps & job.required_caps) == job.required_caps
// ---------------------------------------------------------------------------
enum GpuCap : IUD {
  CapNone         = 0,
  CapFp64         = 1 << 0,  // Real FP64 throughput (V100 yes, 5060 Ti no, CPU no)
  CapVideoEncode  = 1 << 1,  // H.265 10-bit 4:2:2 encode (5060 Ti yes, V100 no)
  CapSim          = 1 << 2,  // Isaac Sim / GL desktop workload (5060 Ti yes)
  CapVectorSearch = 1 << 3,  // Can hold + serve a vector index in its memory
  CapLargeMem     = 1 << 4,  // >= 16 GB addressable (both cards yes, CPU = RAM)
};

// ---------------------------------------------------------------------------
// Runner backend — the thing that actually DRIVES the accelerator.
// Today only "cuda" exists. Stored as a small string tag (not an enum) so
// adding a runner later does not touch this header's matching logic.
// The scheduler routes a job only to an accelerator whose runner it can
// actually drive (i.e. the runner is registered/active).
// ---------------------------------------------------------------------------
enum class RunnerBackend : IUB {
  Unknown  = 0,
  Cuda     = 1,  // The only runner today.
  // Future (do NOT add until the runner exists):
  // Hip      = 2,
  // LevelZero= 3,
  // OpenCL   = 4,
  // Vulkan   = 5,
};

// ---------------------------------------------------------------------------
// Accelerator state — lifecycle.
// ---------------------------------------------------------------------------
enum class AccelState : IUB {
  Free      = 0,  // Available for scheduling
  Allocated = 1,  // Held by a running job
  ColdBoot  = 2,  // Loading vector index RAM->VRAM (GPU-accelerated mode)
  Fault     = 3,  // Error / needs attention
  Offline   = 4,  // Powered down or unreachable
};

// ---------------------------------------------------------------------------
// Job state — lifecycle.
// ---------------------------------------------------------------------------
enum class JobState : IUB {
  Queued  = 0,
  Running = 1,
  Done    = 2,
  Failed  = 3,
};

// ---------------------------------------------------------------------------
// Vector DB mode — orthogonal to the runner. CPU mode needs no accelerator.
// ---------------------------------------------------------------------------
enum class VectorDbMode : IUB {
  Cpu  = 0,  // RAM-resident index, no GPU. Runs on any node, GPU or not.
  Gpu  = 1,  // Index loaded into an accelerator's VRAM for faster search.
};

// ---------------------------------------------------------------------------
// Accelerator — one physical compute device (GPU today, CPU-pool later).
//
// NOTE: "Gpu" is the historical name; the struct is deliberately generic so
// a CPU-only node (vram_total_mib = 0, backend = Cpu) fits the same registry.
// ---------------------------------------------------------------------------
struct Gpu {
  IUD         id;            // Global id (box_id << 8 | local_index)
  ISN         box_id;        // 0 = dev, 1 = prod
  ISN         local_index;   // 0 = V100, 1 = 5060 Ti (local to the box)
  RunnerBackend backend;     // What drives it (Cuda today). Cpu for CPU-pool.
  IUD         caps;          // GpuCap bitmask — the ONLY thing matched on.
  AccelState  state;
  IUC         vram_total_mib;// 0 for CPU-only (uses system RAM instead)
  IUC         vram_used_mib;
  IUC         vram_free_mib() const { return vram_total_mib - vram_used_mib; }
  // A job's required caps are satisfied if the accelerator has ALL of them.
  BOL         HasCaps(IUD required) const {
    return (caps & required) == required;
  }
  CHA*        name;          // "dev-V100", "prod-5060Ti", "dev-cpu-pool", ...
};

// ---------------------------------------------------------------------------
// GpuJob — one unit of work. Keyed by SubsecondId (64-bit, time-ordered, so
// the queue is naturally FIFO-by-submission).
// ---------------------------------------------------------------------------
struct GpuJob {
  IUD        subsecond_id;   // 64-bit SubsecondId — primary key + FIFO order
  IUD        required_caps;  // GpuCap subset the job needs (0 = any)
  IUC        mem_required_mib;// Memory budget (VRAM for GPU, RAM for CPU mode)
  VectorDbMode vector_mode;  // Cpu or Gpu (only meaningful for vector-DB jobs)
  JobState   state;
  IUD        gpu_id;         // 0 = unscheduled, else the Gpu.id it landed on
  ISD        submit_ms;      // Submission timestamp (ms) for accounting
  CHA*       payload;        // Job description / command
};

// ---------------------------------------------------------------------------
// Scheduler status — no exceptions, return codes.
// ---------------------------------------------------------------------------
enum class SchedStatus : ISA {
  Ok          =  0,
  NoAccelFree = -1,  // No accelerator with the required caps is free
  MemTooBig   = -2,  // Accelerator free but memory budget doesn't fit
  BadJob      = -3,  // Null / invalid job pointer
  Full        = -4,  // Queue is full
  NotFound    = -5,  // gpu_id / job_id not found
  NoRunner    = -6,  // Accelerator's backend has no active runner (e.g. CUDA
                     //   not loaded). CPU mode never hits this.
};

// ---------------------------------------------------------------------------
// GpuScheduler — the brain.
//
// Holds the accelerator registry (fixed 2-box x 2-GPU layout today; the
// struct is generic enough to add CPU-pool entries or new boxes) and the
// job queue. All methods return SchedStatus; no exceptions.
//
// BACKEND-AGNOSTIC INvariants:
//   - Matching is on caps, never on backend or card name.
//   - A job is only routed to an accelerator whose backend the scheduler
//     has marked as having an active runner (SetRunnerActive). Today that
//     is Cuda. A CPU-mode vector DB job (required_caps without any GPU-only
//     cap, vector_mode = Cpu) is routable to a CPU-pool entry even when no
//     GPU runner is active.
// ---------------------------------------------------------------------------
class GpuScheduler {
 public:
  // Default constructor — zero all state so `GpuScheduler s;` is safe.
  GpuScheduler()
      : queue_count_(0), accel_count_(0), active_runners_(0), inited_(false) {
    for (ISN i = 0; i < AccelTotal; ++i) {
      accels_[i] = Gpu{};
    }
    for (ISN i = 0; i < JobQueueMax; ++i) {
      queue_[i] = GpuJob{};
    }
  }

  enum {
    BoxCount    = 2,
    AccelPerBox = 2,                  // V100 + 5060 Ti (today)
    AccelTotal  = BoxCount * AccelPerBox,  // 4
    JobQueueMax = 64,
  };

  // Initialize the registry with the standard 2-box x 2-GPU layout.
  // Both cards default to 16384 MiB (16 GB), backend Cuda.
  // V100  caps: CapFp64 | CapVectorSearch | CapLargeMem
  // 5060Ti caps: CapVideoEncode | CapSim | CapVectorSearch | CapLargeMem
  SchedStatus Init(IUC vram_mib = 16384);

  // Mark which runner backends are active (drivable). Today: Cuda.
  // A CPU-pool entry is always "active" (no runner needed).
  void SetRunnerActive(RunnerBackend backend, BOL active);
  BOL RunnerActive(RunnerBackend backend) const;

  // --- Registry queries ---
  const Gpu* AccelById(IUD id) const;
  ISN        AccelCount() const;
  ISN        FreeAccelCount(IUD required_caps) const;
  IUC        FreeMem(IUD required_caps) const;  // Total free mem across matching accels

  // --- Scheduling ---
  // Place the job on a free accelerator that (a) has all required caps,
  // (b) has an active runner (unless the job is CPU-mode), and
  // (c) has enough free memory. Sets job.gpu_id + job.state = Running.
  SchedStatus Schedule(GpuJob& job);

  // Release the accelerator back to Free when the job finishes.
  SchedStatus Unschedule(IUD gpu_id);

  // Mark an accelerator faulted / offline / etc.
  SchedStatus SetGpuState(IUD gpu_id, AccelState state);

  // --- Vector DB cold-boot (GPU-accelerated mode only) ---
  // Begin loading the vector index into the accelerator's VRAM.
  // Returns the RAM->VRAM time budget in ms. CPU mode has no cold-boot
  // (the index is already RAM-resident) and returns 0.
  SchedStatus VectorDbColdBoot(IUD gpu_id, IUC index_size_mib, ISD* budget_ms);

  // The cold-boot target: the GPU-mode index load must complete within this.
  static constexpr ISD ColdBootTargetMs = 1000;  // ~1 second

  // Compute the RAM->VRAM transfer budget for a given index size in MiB.
  // Bottleneck is PCIe Gen4 x16 effective (~25 GB/s), NOT DRAM (130 GB/s).
  // Adds a fixed overhead for CUDA context setup + index deserialization.
  static ISD ColdBootBudgetMs(IUC index_size_mib);

 private:
  Gpu    accels_[AccelTotal];
  GpuJob queue_[JobQueueMax];
  ISN    queue_count_;
  ISN    accel_count_;
  IUD    active_runners_;  // bitmask of RunnerBackend values that are active
  BOL    inited_;
};

}  // namespace _

#endif  // SUBSECONDBD_GPU_SCHEDULER_DECL
