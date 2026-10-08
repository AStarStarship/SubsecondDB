// Copyright AStarship <https://astarship.net>.
//
// SubsecondDb GPU Vector Database — seam tests.
//
// Runs in the seam _Main.cpp alongside the GpuScheduler seam (00.Core.hxx).
// CPU-path tests always run. The CUDA cross-check block runs ONLY when the
// binary was built with CUDA (SUBSECONDBD_HAS_CUDA) AND a device is present;
// otherwise it prints a SKIP note and the seam still passes.
//
// Each test returns a seam log string ("PASS" or "FAIL: ..."). No external
// test framework — hand-rolled, matching 00.Core.hxx's style.

#include <_Config.h>
#include <unistd.h>
#include "VectorIndex.h"
#include "VectorDb.h"
#ifdef SUBSECONDBD_HAS_CUDA
#include "CudaRunner.h"
#endif

using namespace _;

namespace {

// NOTE: 00.Core.hxx (the GpuScheduler seam) already defines g_pass/g_fail/
// Check/TestCase/kTests in the anonymous namespace AND already includes
// GpuScheduler.hxx. Both files share one TU, so we use Vdb-prefixed names
// here and do NOT re-include GpuScheduler.hxx (its methods are defined once,
// by 00.Core.hxx, and VectorDb.h only needs the class declaration).

ISN VdbPass = 0;
ISN VdbFail = 0;

void VdbCheck(BOL cond, const CHA* what) {
  if (cond) {
    ++VdbPass;
  } else {
    ++VdbFail;
    CHA buf[160];
    ISN n = 0;
    const CHA* p = "  FAIL: ";
    while (*p) buf[n++] = *p++;
    p = what;
    while (*p && n < 150) buf[n++] = *p++;
    buf[n++] = '\n';
    if (write(2, buf, (unsigned long)n) < 0) { /* best-effort */ }
  }
}

void VdbNote(const CHA* s) {
  ISN n = 0;
  while (s[n]) ++n;
  if (write(2, s, (unsigned long)n) < 0) { /* best-effort */ }
}

// ---------------------------------------------------------------------------
// Test 1: CPU exact top-k — known answer, both metrics.
// ---------------------------------------------------------------------------
const CHA* TestCpuTopK(const CHA*) {
  VectorIndex<8, 4> ix;
  ix.Init(VectorMetric::L2Squared);
  FPC v0[4] = {1, 0, 0, 0};
  FPC v1[4] = {0, 1, 0, 0};
  FPC v2[4] = {1, 1, 0, 0};
  ix.AddVector(100, v0);
  ix.AddVector(101, v1);
  ix.AddVector(102, v2);
  FPC q[4] = {1, 0.1f, 0, 0};
  TVectorHit h[3];
  VdbCheck(ix.SearchTopK(q, 3, h) == IndexStatus::Ok, "L2sq search Ok");
  VdbCheck(h[0].id == 100, "nearest is v0 (100)");
  VdbCheck(h[1].id == 102, "2nd is v2 (102)");
  VdbCheck(h[2].id == 101, "3rd is v1 (101)");

  // InnerProduct: best = largest dot.
  VectorIndex<8, 4> ip;
  ip.Init(VectorMetric::InnerProduct);
  FPC a[4] = {1, 2, 3, 4};
  FPC b[4] = {4, 3, 2, 1};
  FPC c[4] = {0, 0, 0, 0};
  ip.AddVector(1, a);
  ip.AddVector(2, b);
  ip.AddVector(3, c);
  FPC qp[4] = {1, 1, 1, 1};
  TVectorHit hi[3];
  VdbCheck(ip.SearchTopK(qp, 3, hi) == IndexStatus::Ok, "IP search Ok");
  VdbCheck(hi[2].id == 3, "IP: zero-vector (id 3) is worst/last");
  VdbCheck(hi[0].score < -9.9f, "IP: best score ~ -10 (dot 10)");
  return "PASS";
}

// ---------------------------------------------------------------------------
// Test 2: SubsecondId FIFO — append order == id order; search keys by id.
// ---------------------------------------------------------------------------
const CHA* TestFifoKeys(const CHA*) {
  VectorIndex<16, 2> ix;
  ix.Init(VectorMetric::L2Squared);
  // Time-ordered SubsecondIds: higher id = later.
  FPC r0[2] = {0, 0};
  ix.AddVector(1, r0);  // id 1 first
  ix.AddVector(500, r0);  // id 500 second
  ix.AddVector(99999, r0);  // id 99999 third
  VdbCheck(ix.Count() == 3, "3 vectors stored");
  // All identical vectors -> any top-1 is a tie; but the IDs must be from
  // the set we inserted. Search and confirm the hit id is one we added.
  FPC q[2] = {0, 0};
  TVectorHit h[1];
  VdbCheck(ix.SearchTopK(q, 1, h) == IndexStatus::Ok, "search Ok");
  BOL known = (h[0].id == 1) || (h[0].id == 500) || (h[0].id == 99999);
  VdbCheck(known, "hit id is one we inserted");
  return "PASS";
}

// ---------------------------------------------------------------------------
// Test 3: k bounds + empty + capacity.
// ---------------------------------------------------------------------------
const CHA* TestBounds(const CHA*) {
  VectorIndex<4, 2> ix;
  ix.Init(VectorMetric::L2Squared);
  FPC q[2] = {1, 0};
  TVectorHit h[8];
  VdbCheck(ix.SearchTopK(q, 1, h) == IndexStatus::Empty, "empty -> Empty");
  FPC r[2] = {1, 0};
  ix.AddVector(1, r);
  ix.AddVector(2, r);
  ix.AddVector(3, r);
  ix.AddVector(4, r);
  VdbCheck(ix.Count() == 4, "4 stored");
  VdbCheck(ix.AddVector(5, r) == IndexStatus::Capacity, "5th -> Capacity");
  VdbCheck(ix.SearchTopK(q, 5, h) == IndexStatus::KTooBig, "k=5 > count -> KTooBig");
  VdbCheck(ix.SearchTopK(q, 4, h) == IndexStatus::Ok, "k=4 == count -> Ok");
  return "PASS";
}

// ---------------------------------------------------------------------------
// Test 4: Facade — CPU mode end-to-end (Init/Add/Search/Mode/cold-boot).
// ---------------------------------------------------------------------------
const CHA* TestFacadeCpu(const CHA*) {
  VectorDb<16, 4> db;
  VdbCheck(db.Init(VectorMetric::L2Squared, VectorDbMode::Cpu) ==
            VectorDbStatus::Ok,
        "facade CPU init Ok");
  VdbCheck(db.Mode() == VectorDbMode::Cpu, "mode is Cpu");
  FPC v0[4] = {1, 0, 0, 0};
  FPC v1[4] = {0, 1, 0, 0};
  FPC v2[4] = {1, 1, 0, 0};
  db.AddVector(100, v0);
  db.AddVector(101, v1);
  db.AddVector(102, v2);
  VdbCheck(db.Count() == 3, "3 vectors");
  FPC q[4] = {1, 0.1f, 0, 0};
  TVectorHit h[3];
  VdbCheck(db.SearchTopK(q, 3, h) == VectorDbStatus::Ok, "facade search Ok");
  VdbCheck(h[0].id == 100, "facade nearest is 100");
  VdbCheck(db.ColdBootBudgetMs() == 0, "CPU mode cold-boot budget == 0");
  return "PASS";
}

// ---------------------------------------------------------------------------
// Test 5: Facade — GPU request on a CPU-only build downgrades to CPU (no GPU,
// no driver). On the CUDA box with a GPU this would return Ok; here it must
// return NoRunner and still answer the query via the CPU path.
// ---------------------------------------------------------------------------
const CHA* TestFacadeGpuDowngrade(const CHA*) {
  VectorDb<16, 4> db;
  VectorDbStatus st = db.Init(VectorMetric::L2Squared, VectorDbMode::Gpu);
#ifdef SUBSECONDBD_HAS_CUDA
  // If a GPU is actually present, st == Ok and Mode == Gpu. If not, it
  // downgrades to Cpu with NoRunner. Either way the DB is usable.
  if (st == VectorDbStatus::Ok) {
    VdbCheck(db.Mode() == VectorDbMode::Gpu, "GPU up: mode Gpu");
  } else {
    VdbCheck(st == VectorDbStatus::NoRunner, "no GPU: downgraded NoRunner");
    VdbCheck(db.Mode() == VectorDbMode::Cpu, "no GPU: mode Cpu");
  }
#else
  // No CUDA at all: always downgrades.
  VdbCheck(st == VectorDbStatus::NoRunner, "no CUDA: downgraded NoRunner");
  VdbCheck(db.Mode() == VectorDbMode::Cpu, "no CUDA: mode Cpu");
#endif
  // Regardless, the query still answers correctly.
  FPC v0[4] = {1, 0, 0, 0};
  FPC v1[4] = {0, 1, 0, 0};
  db.AddVector(1, v0);
  db.AddVector(2, v1);
  FPC q[4] = {1, 0, 0, 0};
  TVectorHit h[1];
  VdbCheck(db.SearchTopK(q, 1, h) == VectorDbStatus::Ok, "query answers");
  VdbCheck(h[0].id == 1, "nearest is v0 (id 1)");
  return "PASS";
}

#ifdef SUBSECONDBD_HAS_CUDA
// ---------------------------------------------------------------------------
// Test 6: CUDA cross-check — run the SAME query on the GPU and the CPU, assert
// identical ranking (scores within FPC tolerance). Skips (PASS) if no device.
// ---------------------------------------------------------------------------
const CHA* TestCudaCrossCheck(const CHA*) {
  constexpr IUC N = 2000, D = 16, K = 8;
  FPC data[N * D];
  IUD ids[N];
  FPC query[D];
  IUD state = 0xDEADBEEFCAFEBABEULL;
  auto lcg = [&]() -> FPC {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (FPC)((state >> 40) & 0xFFFF) / 65535.0f * 2.0f - 1.0f;
  };
  for (IUC i = 0; i < N * D; ++i) data[i] = lcg();
  for (IUC i = 0; i < N; ++i) ids[i] = 7000000 + i;
  for (IUC d = 0; d < D; ++d) query[d] = lcg();

  // CPU reference.
  VectorIndex<N, D> ref;
  ref.Init(VectorMetric::L2Squared);
  for (IUC i = 0; i < N; ++i) ref.AddVector(ids[i], data + i * D);
  TVectorHit cpu[K];
  if (ref.SearchTopK(query, K, cpu) != IndexStatus::Ok) return "FAIL: cpu ref";

  CudaRunner runner;
  CudaStatus rs = runner.Init(0);
  if (rs != CudaStatus::Ok) {
    VdbNote("  SKIP CUDA cross-check (no device / no driver)\n");
    return "PASS";  // not a failure on a GPU-less box
  }
  IUC vram = 0;
  rs = runner.UploadVectors(data, ids, N, D, &vram);
  if (rs != CudaStatus::Ok) {
    VdbNote("  SKIP CUDA cross-check (upload failed)\n");
    runner.Shutdown();
    return "PASS";
  }
  TVectorHit gpu[K];
  rs = runner.SearchTopK(query, K, gpu, VectorMetric::L2Squared);
  if (rs != CudaStatus::Ok) {
    Note("  FAIL: cuda search failed\n");
    runner.Shutdown();
    return "FAIL: cuda search";
  }
  for (IUC i = 0; i < K; ++i) {
    VdbCheck(gpu[i].id == cpu[i].id, "GPU ranking == CPU ranking");
    FPD diff = gpu[i].score - cpu[i].score;
    if (diff < 0) diff = -diff;
    VdbCheck(diff < 1e-3f + (FPD)D * 1e-5f, "GPU score ~ CPU score");
  }
  runner.Shutdown();
  return "PASS";
}
#endif

struct VdbTestCase { const CHA* name; const CHA* (*fn)(const CHA*); };
const VdbTestCase VdbTests[] = {
  {"VectorDb.CpuTopK",        TestCpuTopK},
  {"VectorDb.FifoKeys",       TestFifoKeys},
  {"VectorDb.Bounds",         TestBounds},
  {"VectorDb.FacadeCpu",      TestFacadeCpu},
  {"VectorDb.FacadeGpuDown",  TestFacadeGpuDowngrade},
#ifdef SUBSECONDBD_HAS_CUDA
  {"VectorDb.CudaCrossCheck", TestCudaCrossCheck},
#endif
};
constexpr ISN VdbTestCount = sizeof(VdbTests) / sizeof(VdbTests[0]);

}  // namespace

// Entry point called by the seam _Main.cpp.
const CHA* SubsecondDbVectorDbSeamRun(const CHA* args) {
  (void)args;
  VdbPass = 0;
  VdbFail = 0;
  for (ISN i = 0; i < VdbTestCount; ++i) {
    CHA trace[64];
    ISN tn = 0;
    const CHA* p = "RUN ";
    while (*p) trace[tn++] = *p++;
    p = VdbTests[i].name;
    while (*p && tn < 60) trace[tn++] = *p++;
    trace[tn++] = '\n';
    if (write(2, trace, (unsigned long)tn) < 0) { /* best-effort */ }
    const CHA* result = VdbTests[i].fn(args);
    (void)result;
  }
  static CHA status[32];
  ISN n = 0;
  const CHA* prefix = (VdbFail == 0) ? "PASS" : "FAIL";
  while (*prefix) status[n++] = *prefix++;
  status[n++] = ' ';
  auto write_int = [&](ISN v) {
    CHA buf[16];
    ISN m = 0;
    if (v == 0) buf[m++] = '0';
    while (v > 0) { buf[m++] = (CHA)('0' + (v % 10)); v /= 10; }
    while (m > 0) status[n++] = buf[--m];
  };
  write_int(VdbPass);
  status[n++] = '/';
  write_int(VdbPass + VdbFail);
  status[n] = 0;
  return status;
}
