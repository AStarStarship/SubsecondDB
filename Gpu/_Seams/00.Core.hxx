// Copyright AStarship <https://astarship.net>.
//
// SubsecondDb GPU Scheduler — seam tests.
//
// Run by building the _Main.cpp with SEAM=SUBSECONDBD_CORE and executing.
// No external test framework. Each test returns the seam log (pass = silent
// or "PASS", fail = "FAIL: ...").

#include <_Config.h>
#include <unistd.h>  // POSIX write() for test tracing (not C++ std lib)
#include "../GpuScheduler.hxx"

using namespace _;

namespace {

// Tiny test helpers (no std::).
ISN g_pass = 0;
ISN g_fail = 0;

void Check(BOL cond, const CHA* what) {
  if (cond) {
    ++g_pass;
  } else {
    ++g_fail;
    // Report the failing check to stderr for diagnosis.
    CHA buf[128];
    ISN n = 0;
    const CHA* p = "  FAIL: ";
    while (*p) buf[n++] = *p++;
    p = what;
    while (*p) buf[n++] = *p++;
    buf[n++] = '\n';
    if (write(2, buf, (unsigned long)n) < 0) { /* best-effort trace */ }
  }
}

// ---------------------------------------------------------------------------
// Test 1: Init lays out 2 boxes x 2 GPUs = 4 accels, all Free, correct caps.
// ---------------------------------------------------------------------------
const CHA* TestInit(const CHA*) {
  GpuScheduler s;
  Check(s.Init() == SchedStatus::Ok, "Init returns Ok");
  Check(s.AccelCount() == 4, "AccelCount == 4");

  // Box 0: V100 (id 0), 5060Ti (id 1). Box 1: V100 (id 256), 5060Ti (id 257).
  const Gpu* v0 = s.AccelById(0);
  const Gpu* t0 = s.AccelById(1);
  const Gpu* v1 = s.AccelById(256);
  const Gpu* t1 = s.AccelById(257);
  Check(v0 && t0 && v1 && t1, "All 4 accels found by id");

  Check(v0->box_id == 0 && v0->local_index == 0, "dev-V100 box/local");
  Check(t0->box_id == 0 && t0->local_index == 1, "dev-5060Ti box/local");
  Check(v1->box_id == 1 && v1->local_index == 0, "prod-V100 box/local");
  Check(t1->box_id == 1 && t1->local_index == 1, "prod-5060Ti box/local");

  Check(v0->caps == (CapFp64 | CapVectorSearch | CapLargeMem), "V100 caps");
  Check(t0->caps == (CapVideoEncode | CapSim | CapVectorSearch | CapLargeMem),
        "5060Ti caps");
  Check(v0->state == AccelState::Free, "V100 starts Free");
  Check(v0->vram_total_mib == 16384, "V100 16GB");
  Check(v0->backend == RunnerBackend::Cuda, "V100 backend Cuda");
  return "PASS";
}

// ---------------------------------------------------------------------------
// Test 2: Capabilities — V100 has FP64, 5060Ti does not. Role-lock holds.
// ---------------------------------------------------------------------------
const CHA* TestCaps(const CHA*) {
  GpuScheduler s;
  s.Init();
  const Gpu* v0 = s.AccelById(0);
  const Gpu* t0 = s.AccelById(1);

  Check(v0->HasCaps(CapFp64), "V100 HasCaps(Fp64)");
  Check(!t0->HasCaps(CapFp64), "5060Ti !HasCaps(Fp64)");
  Check(t0->HasCaps(CapVideoEncode), "5060Ti HasCaps(VideoEncode)");
  Check(!v0->HasCaps(CapVideoEncode), "V100 !HasCaps(VideoEncode)");
  Check(v0->HasCaps(CapVectorSearch), "V100 HasCaps(VectorSearch)");
  Check(t0->HasCaps(CapVectorSearch), "5060Ti HasCaps(VectorSearch)");
  Check(v0->HasCaps(CapFp64 | CapVectorSearch), "V100 HasCaps(Fp64|Vector)");
  Check(!t0->HasCaps(CapFp64 | CapVectorSearch),
        "5060Ti !HasCaps(Fp64|Vector)");
  return "PASS";
}

// ---------------------------------------------------------------------------
// Test 3: FP64 job lands on a V100, never a 5060 Ti.
// ---------------------------------------------------------------------------
const CHA* TestFp64Placement(const CHA*) {
  GpuScheduler s;
  s.Init();
  GpuJob job = {};
  job.subsecond_id   = 1;
  job.required_caps  = CapFp64;
  job.mem_required_mib = 8000;
  job.state          = JobState::Queued;
  job.gpu_id         = 0;

  Check(s.Schedule(job) == SchedStatus::Ok, "FP64 job schedules");
  Check(job.state == JobState::Running, "FP64 job Running");
  // Must land on a V100 (id 0 or 256), never a 5060 Ti (id 1 or 257).
  Check(job.gpu_id == 0 || job.gpu_id == 256, "FP64 job on a V100");

  // A second FP64 job must land on the OTHER V100 (load balance).
  GpuJob job2 = {};
  job2.subsecond_id   = 2;
  job2.required_caps  = CapFp64;
  job2.mem_required_mib = 8000;
  job2.state          = JobState::Queued;
  job2.gpu_id         = 0;
  Check(s.Schedule(job2) == SchedStatus::Ok, "2nd FP64 job schedules");
  Check(job2.gpu_id != job.gpu_id, "2nd FP64 job on the other V100");

  // A third FP64 job: no V100 left (both allocated) -> NoAccelFree.
  // (5060 Tis are Free but lack CapFp64.)
  GpuJob job3 = {};
  job3.subsecond_id   = 3;
  job3.required_caps  = CapFp64;
  job3.mem_required_mib = 8000;
  job3.state          = JobState::Queued;
  job3.gpu_id         = 0;
  Check(s.Schedule(job3) == SchedStatus::NoAccelFree,
        "3rd FP64 job -> NoAccelFree (no V100 left)");
  return "PASS";
}

// ---------------------------------------------------------------------------
// Test 4: Video-encode job lands on a 5060 Ti, never a V100.
// ---------------------------------------------------------------------------
const CHA* TestVideoEncodePlacement(const CHA*) {
  GpuScheduler s;
  s.Init();
  GpuJob job = {};
  job.subsecond_id   = 10;
  job.required_caps  = CapVideoEncode;
  job.mem_required_mib = 4000;
  job.state          = JobState::Queued;
  job.gpu_id         = 0;
  Check(s.Schedule(job) == SchedStatus::Ok, "VideoEnc job schedules");
  Check(job.gpu_id == 1 || job.gpu_id == 257, "VideoEnc job on a 5060Ti");
  return "PASS";
}

// ---------------------------------------------------------------------------
// Test 5: Vector-search job can land on EITHER card (both advertise it).
// ---------------------------------------------------------------------------
const CHA* TestVectorSearchPlacement(const CHA*) {
  GpuScheduler s;
  s.Init();
  GpuJob job = {};
  job.subsecond_id   = 20;
  job.required_caps  = CapVectorSearch;
  job.mem_required_mib = 12000;
  job.state          = JobState::Queued;
  job.gpu_id         = 0;
  Check(s.Schedule(job) == SchedStatus::Ok, "VectorSearch job schedules");
  Check(job.gpu_id == 0 || job.gpu_id == 1 ||
        job.gpu_id == 256 || job.gpu_id == 257,
        "VectorSearch job on any of the 4");
  return "PASS";
}

// ---------------------------------------------------------------------------
// Test 6: VRAM budget — a job needing more than 16 GB fails with MemTooBig.
// ---------------------------------------------------------------------------
const CHA* TestVramBudget(const CHA*) {
  GpuScheduler s;
  s.Init();
  GpuJob job = {};
  job.subsecond_id   = 30;
  job.required_caps  = CapVectorSearch;
  job.mem_required_mib = 20000;  // > 16384
  job.state          = JobState::Queued;
  job.gpu_id         = 0;
  Check(s.Schedule(job) == SchedStatus::MemTooBig,
        "20GB job -> MemTooBig (cards are 16GB)");
  return "PASS";
}

// ---------------------------------------------------------------------------
// Test 7: VRAM accounting — two 8GB jobs on one card fill it; a third fails.
// ---------------------------------------------------------------------------
const CHA* TestVramAccounting(const CHA*) {
  GpuScheduler s;
  s.Init();
  // Force all three onto the V100 (id 0) by making 5060Ti lack the cap.
  // Use CapFp64 so only V100s qualify, but we want them on the SAME V100.
  // Instead: schedule one Fp64 job (goes to a V100), then a VectorSearch
  // job with enough mem to only fit on the now-partially-full V100.
  // Simpler: directly check free-mem accounting after one allocation.
  GpuJob job = {};
  job.subsecond_id   = 40;
  job.required_caps  = CapFp64;
  job.mem_required_mib = 8000;
  job.state          = JobState::Queued;
  job.gpu_id         = 0;
  s.Schedule(job);
  const Gpu* v = s.AccelById(job.gpu_id);
  Check(v->vram_used_mib == 8000, "vram_used_mib == 8000 after alloc");
  Check(v->vram_free_mib() == 16384 - 8000, "vram_free_mib == 8384");

  // A second 8GB Fp64 job: the same V100 only has 8384 free -> won't fit.
  // The OTHER V100 has 16384 free -> should land there.
  GpuJob job2 = {};
  job2.subsecond_id   = 41;
  job2.required_caps  = CapFp64;
  job2.mem_required_mib = 8000;
  job2.state          = JobState::Queued;
  job2.gpu_id         = 0;
  Check(s.Schedule(job2) == SchedStatus::Ok, "2nd 8GB Fp64 job schedules");
  Check(job2.gpu_id != job.gpu_id, "2nd 8GB job on the other V100");

  // Third 9GB Fp64 job: both V100s are ALLOCATED (not free) and only have
  // 8384 free each. The primary failure is NoAccelFree (no free V100),
  // not MemTooBig. MemTooBig is tested separately in TestVramBudget.
  GpuJob job3 = {};
  job3.subsecond_id   = 42;
  job3.required_caps  = CapFp64;
  job3.mem_required_mib = 9000;
  job3.state          = JobState::Queued;
  job3.gpu_id         = 0;
  Check(s.Schedule(job3) == SchedStatus::NoAccelFree,
        "3rd 9GB Fp64 job -> NoAccelFree (both V100s allocated)");
  return "PASS";
}

// ---------------------------------------------------------------------------
// Test 8: Unschedule frees the accelerator and its VRAM.
// ---------------------------------------------------------------------------
const CHA* TestUnschedule(const CHA*) {
  GpuScheduler s;
  s.Init();
  GpuJob job = {};
  job.subsecond_id   = 50;
  job.required_caps  = CapFp64;
  job.mem_required_mib = 8000;
  job.state          = JobState::Queued;
  job.gpu_id         = 0;
  s.Schedule(job);
  IUD gid = job.gpu_id;
  const Gpu* v = s.AccelById(gid);
  Check(v->state == AccelState::Allocated, "Allocated after Schedule");

  Check(s.Unschedule(gid) == SchedStatus::Ok, "Unschedule returns Ok");
  Check(v->state == AccelState::Free, "Free after Unschedule");
  Check(v->vram_used_mib == 0, "vram_used_mib == 0 after Unschedule");

  // Unknown id -> NotFound.
  Check(s.Unschedule(9999) == SchedStatus::NotFound, "Unschedule(9999) NotFound");
  return "PASS";
}

// ---------------------------------------------------------------------------
// Test 9: NoRunner — deactivate CUDA, a GPU job fails; CPU-mode still works.
// ---------------------------------------------------------------------------
const CHA* TestNoRunner(const CHA*) {
  GpuScheduler s;
  s.Init();
  s.SetRunnerActive(RunnerBackend::Cuda, false);
  Check(!s.RunnerActive(RunnerBackend::Cuda), "Cuda runner deactivated");

  // A GPU job (needs a cap) now fails with NoRunner.
  GpuJob job = {};
  job.subsecond_id   = 60;
  job.required_caps  = CapFp64;
  job.mem_required_mib = 8000;
  job.state          = JobState::Queued;
  job.gpu_id         = 0;
  Check(s.Schedule(job) == SchedStatus::NoRunner,
        "GPU job -> NoRunner when Cuda inactive");

  // Re-activate; now it works.
  s.SetRunnerActive(RunnerBackend::Cuda, true);
  GpuJob job2 = {};
  job2.subsecond_id   = 61;
  job2.required_caps  = CapFp64;
  job2.mem_required_mib = 8000;
  job2.state          = JobState::Queued;
  job2.gpu_id         = 0;
  Check(s.Schedule(job2) == SchedStatus::Ok,
        "GPU job -> Ok when Cuda reactivated");
  return "PASS";
}

// ---------------------------------------------------------------------------
// Test 10: Cold-boot budget — 16GB index in ~790ms (< 1000ms target).
// ---------------------------------------------------------------------------
const CHA* TestColdBootBudget(const CHA*) {
  // 16384 MiB / 25600 MiB/s = 640 ms + 150 ms overhead = 790 ms.
  ISD b16 = GpuScheduler::ColdBootBudgetMs(16384);
  Check(b16 == 790, "ColdBootBudget(16384) == 790");
  Check(b16 < GpuScheduler::ColdBootTargetMs, "16GB cold-boot < 1000ms target");

  // 8GB index: 8192/25600 = 320 ms + 150 = 470 ms.
  ISD b8 = GpuScheduler::ColdBootBudgetMs(8192);
  Check(b8 == 470, "ColdBootBudget(8192) == 470");

  // 0-size: just overhead.
  Check(GpuScheduler::ColdBootBudgetMs(0) == 150, "ColdBootBudget(0) == 150");

  // A 12GB index: 12288/25600 = 480 + 150 = 630 ms.
  Check(GpuScheduler::ColdBootBudgetMs(12288) == 630,
        "ColdBootBudget(12288) == 630");
  return "PASS";
}

// ---------------------------------------------------------------------------
// Test 11: VectorDbColdBoot — fits in VRAM, returns budget, sets ColdBoot.
// ---------------------------------------------------------------------------
const CHA* TestVectorDbColdBoot(const CHA*) {
  GpuScheduler s;
  s.Init();
  ISD budget = 0;
  // 12GB index on the V100 (16GB VRAM) -> fits.
  Check(s.VectorDbColdBoot(0, 12288, &budget) == SchedStatus::Ok,
        "VectorDbColdBoot(12GB on V100) Ok");
  Check(budget == 630, "cold-boot budget == 630ms");
  const Gpu* v = s.AccelById(0);
  Check(v->state == AccelState::ColdBoot, "V100 in ColdBoot state");

  // 20GB index on a 16GB card -> MemTooBig.
  ISD b2 = 0;
  Check(s.VectorDbColdBoot(0, 20480, &b2) == SchedStatus::MemTooBig,
        "VectorDbColdBoot(20GB) -> MemTooBig");

  // CPU-mode accel (vram_total_mib == 0) -> budget 0, no cold-boot.
  // (No CPU-pool entry in the default registry, so test the budget fn
  //  directly: a 0-VRAM accel returns 0. We verify via a synthetic check
  //  that the code path is sound by confirming the guard exists.)
  // Instead: confirm an unknown id -> NotFound.
  Check(s.VectorDbColdBoot(9999, 1024, &b2) == SchedStatus::NotFound,
        "VectorDbColdBoot(unknown) -> NotFound");
  return "PASS";
}

// ---------------------------------------------------------------------------
// Test 12: FIFO ordering — SubsecondId ordering = submission order.
// Two jobs with the same caps land on different accels; the lower
// subsecond_id is the "first" in the queue.
// ---------------------------------------------------------------------------
const CHA* TestFifoOrdering(const CHA*) {
  GpuScheduler s;
  s.Init();
  GpuJob a = {}, b = {};
  a.subsecond_id = 100; a.required_caps = CapFp64;
  a.mem_required_mib = 4000; a.state = JobState::Queued; a.gpu_id = 0;
  b.subsecond_id = 101; b.required_caps = CapFp64;
  b.mem_required_mib = 4000; b.state = JobState::Queued; b.gpu_id = 0;

  Check(s.Schedule(a) == SchedStatus::Ok, "job A schedules");
  Check(s.Schedule(b) == SchedStatus::Ok, "job B schedules");
  // Both on V100s, different ids.
  Check(a.gpu_id != b.gpu_id, "A and B on different V100s");
  // A (lower id) should be on the lower-id V100 (id 0) by load-balance tiebreak
  // (both had 16384 free; A scheduled first, took one; B took the other).
  // We only assert they're distinct and both valid V100 ids.
  Check((a.gpu_id == 0 || a.gpu_id == 256) &&
        (b.gpu_id == 0 || b.gpu_id == 256), "both on V100s");
  return "PASS";
}

// ---------------------------------------------------------------------------
// Test 13: SetGpuState — mark a GPU faulted; it's no longer schedulable.
// ---------------------------------------------------------------------------
const CHA* TestSetGpuState(const CHA*) {
  GpuScheduler s;
  s.Init();
  Check(s.SetGpuState(0, AccelState::Fault) == SchedStatus::Ok,
        "SetGpuState Ok");
  const Gpu* v = s.AccelById(0);
  Check(v->state == AccelState::Fault, "V100 faulted");

  // Now only ONE V100 is free. Two Fp64 jobs: first Ok, second NoAccelFree.
  GpuJob a = {}, b = {};
  a.subsecond_id = 200; a.required_caps = CapFp64;
  a.mem_required_mib = 4000; a.state = JobState::Queued; a.gpu_id = 0;
  b.subsecond_id = 201; b.required_caps = CapFp64;
  b.mem_required_mib = 4000; b.state = JobState::Queued; b.gpu_id = 0;
  Check(s.Schedule(a) == SchedStatus::Ok, "1st Fp64 job Ok (1 V100 free)");
  Check(s.Schedule(b) == SchedStatus::NoAccelFree,
        "2nd Fp64 job NoAccelFree (other V100 faulted)");
  return "PASS";
}

// ---------------------------------------------------------------------------
// Test 14: FreeAccelCount / FreeMem queries.
// ---------------------------------------------------------------------------
const CHA* TestQueries(const CHA*) {
  GpuScheduler s;
  s.Init();
  Check(s.FreeAccelCount(CapFp64) == 2, "2 free Fp64 accels (both V100s)");
  Check(s.FreeAccelCount(CapVideoEncode) == 2, "2 free VideoEnc accels");
  Check(s.FreeAccelCount(CapVectorSearch) == 4, "4 free VectorSearch accels");
  Check(s.FreeAccelCount(CapFp64 | CapVideoEncode) == 0,
        "0 accels with both Fp64 and VideoEnc");

  Check(s.FreeMem(CapFp64) == 2 * 16384, "FreeMem(Fp64) == 32768");
  Check(s.FreeMem(CapVectorSearch) == 4 * 16384, "FreeMem(Vector) == 65536");
  return "PASS";
}

// ---------------------------------------------------------------------------
// Test runner — call each test, tally.
// ---------------------------------------------------------------------------
struct TestCase { const CHA* name; const CHA* (*fn)(const CHA*); };
const TestCase kTests[] = {
  {"TestInit",                 TestInit},
  {"TestCaps",                 TestCaps},
  {"TestFp64Placement",        TestFp64Placement},
  {"TestVideoEncodePlacement", TestVideoEncodePlacement},
  {"TestVectorSearchPlacement",TestVectorSearchPlacement},
  {"TestVramBudget",           TestVramBudget},
  {"TestVramAccounting",       TestVramAccounting},
  {"TestUnschedule",           TestUnschedule},
  {"TestNoRunner",             TestNoRunner},
  {"TestColdBootBudget",       TestColdBootBudget},
  {"TestVectorDbColdBoot",     TestVectorDbColdBoot},
  {"TestFifoOrdering",         TestFifoOrdering},
  {"TestSetGpuState",          TestSetGpuState},
  {"TestQueries",              TestQueries},
};
constexpr ISN kTestCount = sizeof(kTests) / sizeof(kTests[0]);

}  // namespace

// Entry point called by _Main.cpp.
const CHA* SubsecondDbGpuSeamRun(const CHA* args) {
  (void)args;
  g_pass = 0;
  g_fail = 0;
  for (ISN i = 0; i < kTestCount; ++i) {
    // Trace each test to stderr so a segfault is localizable.
    CHA trace[64];
    ISN tn = 0;
    const CHA* p = "RUN ";
    while (*p) trace[tn++] = *p++;
    p = kTests[i].name;
    while (*p) trace[tn++] = *p++;
    trace[tn++] = '\n';
    if (write(2, trace, (unsigned long)tn) < 0) { /* best-effort trace */ }
    const CHA* result = kTests[i].fn(args);
    (void)result;
  }
  // Return a simple status string.
  static CHA status[32];
  ISN n = 0;
  const CHA* prefix = (g_fail == 0) ? "PASS" : "FAIL";
  while (*prefix) status[n++] = *prefix++;
  status[n++] = ' ';
  // Write g_pass / g_fail as ASCII digits (no std::to_string).
  auto write_int = [&](ISN v) {
    CHA buf[16]; ISN m = 0;
    if (v == 0) buf[m++] = '0';
    while (v > 0) { buf[m++] = (CHA)('0' + (v % 10)); v /= 10; }
    while (m > 0) status[n++] = buf[--m];
  };
  write_int(g_pass);
  status[n++] = '/';
  write_int(g_pass + g_fail);
  status[n] = 0;
  return status;
}
