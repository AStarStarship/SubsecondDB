// Copyright AStarship <https://astarship.net>.
#include "GpuScheduler.h"

namespace _ {

// ---------------------------------------------------------------------------
// Default capability sets for the standard 2-card-per-box layout.
// These are DATA, not logic — they encode what each physical card advertises.
// A new card = a new row here, no scheduler logic change.
// ---------------------------------------------------------------------------
namespace {
constexpr IUD kV100Caps  = CapFp64 | CapVectorSearch | CapLargeMem;
constexpr IUD k5060TiCaps = CapVideoEncode | CapSim | CapVectorSearch | CapLargeMem;

// PCIe Gen4 x16 effective bandwidth ~25 GB/s = 25600 MiB/s.
// Fixed overhead: CUDA context setup + index deserialization + first-touch.
constexpr IUD kPcieMiBps     = 25600;  // MiB per second
constexpr ISD kOverheadMs    = 150;    // ms, fixed context/deser overhead
}  // namespace

// ---------------------------------------------------------------------------
SchedStatus GpuScheduler::Init(IUC vram_mib) {
  if (inited_) return SchedStatus::Ok;  // idempotent
  inited_   = true;
  queue_count_ = 0;
  accel_count_ = 0;
  active_runners_ = 0;

  // 2 boxes x 2 GPUs. Box 0 = dev, Box 1 = prod.
  // local 0 = V100, local 1 = 5060 Ti.
  for (ISN box = 0; box < BoxCount; ++box) {
    // V100
    Gpu& v = accels_[accel_count_++];
    v.id             = (IUD(box) << 8) | 0;
    v.box_id         = box;
    v.local_index    = 0;
    v.backend        = RunnerBackend::Cuda;
    v.caps           = kV100Caps;
    v.state          = AccelState::Free;
    v.vram_total_mib = vram_mib;
    v.vram_used_mib  = 0;
    v.name           = nullptr;  // set by caller or left null

    // 5060 Ti
    Gpu& t = accels_[accel_count_++];
    t.id             = (IUD(box) << 8) | 1;
    t.box_id         = box;
    t.local_index    = 1;
    t.backend        = RunnerBackend::Cuda;
    t.caps           = k5060TiCaps;
    t.state          = AccelState::Free;
    t.vram_total_mib = vram_mib;
    t.vram_used_mib  = 0;
    t.name           = nullptr;
  }

  // Cuda is the only runner, and it is active by default.
  active_runners_ |= (IUD)1 << (IUD)RunnerBackend::Cuda;
  return SchedStatus::Ok;
}

// ---------------------------------------------------------------------------
void GpuScheduler::SetRunnerActive(RunnerBackend backend, BOL active) {
  IUD bit = (IUD)1 << (IUD)backend;
  if (active)      active_runners_ |= bit;
  else             active_runners_ &= ~bit;
}

BOL GpuScheduler::RunnerActive(RunnerBackend backend) const {
  return (active_runners_ & ((IUD)1 << (IUD)backend)) != 0;
}

// ---------------------------------------------------------------------------
const Gpu* GpuScheduler::AccelById(IUD id) const {
  for (ISN i = 0; i < accel_count_; ++i) {
    if (accels_[i].id == id) return &accels_[i];
  }
  return nullptr;
}

ISN GpuScheduler::AccelCount() const { return accel_count_; }

// ---------------------------------------------------------------------------
ISN GpuScheduler::FreeAccelCount(IUD required_caps) const {
  ISN n = 0;
  for (ISN i = 0; i < accel_count_; ++i) {
    const Gpu& g = accels_[i];
    if (g.state != AccelState::Free) continue;
    if (!g.HasCaps(required_caps)) continue;
    // CPU-mode jobs (no GPU-only cap required) can use any free accel incl. CPU-pool.
    // GPU-runner jobs need an active runner for the accel's backend.
    if (required_caps != 0 && !RunnerActive(g.backend)) continue;
    ++n;
  }
  return n;
}

IUC GpuScheduler::FreeMem(IUD required_caps) const {
  IUC total = 0;
  for (ISN i = 0; i < accel_count_; ++i) {
    const Gpu& g = accels_[i];
    if (g.state != AccelState::Free) continue;
    if (!g.HasCaps(required_caps)) continue;
    if (required_caps != 0 && !RunnerActive(g.backend)) continue;
    total += g.vram_free_mib();
  }
  return total;
}

// ---------------------------------------------------------------------------
// Placement: find the first free accelerator that
//   (a) has all required caps,
//   (b) has an active runner (unless the job needs no GPU at all), and
//   (c) has enough free memory for the job's budget.
// Ties broken by: prefer the box with more total free memory (load balance),
// then lower id.
// ---------------------------------------------------------------------------
SchedStatus GpuScheduler::Schedule(GpuJob& job) {
  if (job.state == JobState::Running || job.state == JobState::Done)
    return SchedStatus::BadJob;
  if (queue_count_ >= JobQueueMax) return SchedStatus::Full;

  const IUD need = job.required_caps;
  const BOL needs_runner = (need != 0);  // any cap requirement implies a real accel

  ISN best = -1;
  IUC best_free = 0;
  for (ISN i = 0; i < accel_count_; ++i) {
    const Gpu& g = accels_[i];
    if (g.state != AccelState::Free) continue;
    if (!g.HasCaps(need)) continue;
    if (needs_runner && !RunnerActive(g.backend)) continue;
    if (g.vram_free_mib() < job.mem_required_mib) continue;
    // Load-balance: prefer the accel with the most free memory.
    if (best == -1 || g.vram_free_mib() > best_free) {
      best = i;
      best_free = g.vram_free_mib();
    }
  }

  if (best == -1) {
    // Distinguish "no accel with caps" vs "no runner" vs "mem too big".
    ISN with_caps = 0, with_runner = 0, with_mem = 0;
    for (ISN i = 0; i < accel_count_; ++i) {
      const Gpu& g = accels_[i];
      if (g.state != AccelState::Free) continue;
      if (!g.HasCaps(need)) continue;
      ++with_caps;
      if (needs_runner && !RunnerActive(g.backend)) continue;
      ++with_runner;
      if (g.vram_free_mib() < job.mem_required_mib) continue;
      ++with_mem;
    }
    if (with_caps == 0)      return SchedStatus::NoAccelFree;
    if (with_runner == 0)    return SchedStatus::NoRunner;
    if (with_mem == 0)       return SchedStatus::MemTooBig;
    return SchedStatus::NoAccelFree;
  }

  Gpu& g = accels_[best];
  g.state         = AccelState::Allocated;
  g.vram_used_mib += job.mem_required_mib;
  job.gpu_id      = g.id;
  job.state       = JobState::Running;

  // Enqueue for tracking.
  queue_[queue_count_++] = job;
  return SchedStatus::Ok;
}

// ---------------------------------------------------------------------------
SchedStatus GpuScheduler::Unschedule(IUD gpu_id) {
  const Gpu* g = AccelById(gpu_id);
  if (!g) return SchedStatus::NotFound;
  if (g->state != AccelState::Allocated && g->state != AccelState::ColdBoot)
    return SchedStatus::NotFound;
  // Find the job holding it and mark done.
  for (ISN i = 0; i < queue_count_; ++i) {
    if (queue_[i].gpu_id == gpu_id && queue_[i].state == JobState::Running) {
      queue_[i].state = JobState::Done;
      break;
    }
  }
  Gpu* gm = const_cast<Gpu*>(g);
  gm->vram_used_mib = 0;
  gm->state = AccelState::Free;
  return SchedStatus::Ok;
}

// ---------------------------------------------------------------------------
SchedStatus GpuScheduler::SetGpuState(IUD gpu_id, AccelState state) {
  const Gpu* g = AccelById(gpu_id);
  if (!g) return SchedStatus::NotFound;
  // NOTE: accels_ is mutable here; const_cast is safe because Init owns the
  // storage and no other pointer is handed out as mutable.
  const_cast<Gpu*>(g)->state = state;
  return SchedStatus::Ok;
}

// ---------------------------------------------------------------------------
SchedStatus GpuScheduler::VectorDbColdBoot(IUD gpu_id, IUC index_size_mib,
                                           ISD* budget_ms) {
  const Gpu* g = AccelById(gpu_id);
  if (!g) return SchedStatus::NotFound;
  // CPU mode: index is already RAM-resident, no cold-boot.
  if (g->vram_total_mib == 0) {
    if (budget_ms) *budget_ms = 0;
    return SchedStatus::Ok;
  }
  // GPU mode: must fit in VRAM.
  if (index_size_mib > g->vram_total_mib) return SchedStatus::MemTooBig;
  if (!RunnerActive(g->backend))          return SchedStatus::NoRunner;

  ISD budget = ColdBootBudgetMs(index_size_mib);
  if (budget_ms) *budget_ms = budget;

  const_cast<Gpu*>(g)->state = AccelState::ColdBoot;
  return SchedStatus::Ok;
}

// ---------------------------------------------------------------------------
// RAM->VRAM budget. Bottleneck = PCIe Gen4 x16 (~25 GB/s effective),
// NOT DRAM (130 GB/s). transfer_ms = size / bandwidth, + fixed overhead.
// Example: 16384 MiB (16 GB) / 25600 MiB/s = 640 ms + 150 ms = 790 ms < 1000.
// ---------------------------------------------------------------------------
ISD GpuScheduler::ColdBootBudgetMs(IUC index_size_mib) {
  if (index_size_mib == 0) return kOverheadMs;
  ISD transfer_ms = (ISD)index_size_mib * 1000 / (ISD)kPcieMiBps;
  return transfer_ms + kOverheadMs;
}

}  // namespace _
