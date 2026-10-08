# SubsecondDb GPU Vector Database

GPU-accelerated vector database for the iGeek training loop, built on the
existing `GpuScheduler` (placement/registry) and the ASCIICrabs / CrabsTK type
system. Bare-metal `<cuda_runtime.h>` for the runner — no CUDA toolkit libs,
no thrust/cub/cuBLAS.

## What's here

| File | Role |
|---|---|
| `GpuScheduler.h/.hxx` | The brain: accelerator registry, VRAM budget, cold-boot prediction. Backend-agnostic (pre-existing). |
| `VectorIndex.h` | Flat exact ANN index. `FPC` (float32) rows keyed by `IUD` SubsecondId, L2sq + inner-product exact top-k, CPU reference search. No C++ std lib. |
| `CudaRunner.h/.cu` | The body: bare-metal CUDA runner. `__global__` kernels (score + 2-stage top-k). The ONLY file that touches `<cuda_runtime.h>`. Compiled by nvcc only. |
| `VectorDb.h` | The facade the Pg/ extension and iGeek training loop talk to. Wires runner + index + scheduler. CPU/GPU runtime mode. CrabsTK `TRecord`/`TReader`/`TWriter` corpus I/O. |
| `_Seams/00.Core.hxx` | GpuScheduler seam tests (pre-existing, 68 checks). |
| `_Seams/01.VectorDb.hxx` | VectorDb seam tests (CPU path always; CUDA cross-check when built with CUDA + device present). |
| `_Seams/01.VectorDbCrossCheck.cpp` | Standalone CPU proof that the CUDA runner's 2-stage top-k ALGORITHM (mirrored host-side) produces the same ranking as the CPU reference. Runs without a GPU. |
| `CMakeLists.txt` | Two targets: `subseconddb-gpu` (CPU, always) and `subseconddb-gpu-cuda` (only if nvcc found). |

## Modes

- **CPU mode** — RAM-resident index, CPU reference search. Runs on any node,
  GPU or not. This is the default and the fallback.
- **GPU mode** — index loaded into VRAM, search runs the CUDA kernels. If the
  CUDA runner can't `Init()` (no GPU / no driver), the facade **downgrades to
  CPU mode and returns `NoRunner`** so a query still answers correctly. A
  per-query upload failure also falls back to CPU.

GPU mode is a **runtime choice, not a hardware lock** (per the design
constraints). Swapping the CUDA runner for HIP/LevelZero/OpenCL later is a
runner replacement, not a scheduler rewrite — the scheduler matches on
capabilities, never on the runner.

## Build

### CPU build (any box, no CUDA needed)

```sh
cd SubsecondDb/Gpu
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/subseconddb-gpu        # runs both seams; expect "PASS 68/68 PASS 25/25"
```

The CUDA target is skipped with a message when `nvcc` is absent; the CPU
target is never blocked.

### CUDA build (the box with the V100 / 5060 Ti)

Requirements: NVIDIA driver loaded (`nvidia-smi` works), CUDA toolkit on
`PATH` (`nvcc`). The box: Xeon Gold 6230, 96 GB RAM, 1x V100 16 GB (sm_70),
1x RTX 5060 Ti 16 GB (sm_89).

```sh
cd SubsecondDb/Gpu
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="70 89"   # V100 + 5060 Ti; or 'native'
cmake --build build
./build/subseconddb-gpu-cuda      # runs both seams incl. the live GPU cross-check
```

`subseconddb-gpu-cuda` defines `SUBSECONDBD_HAS_CUDA`, compiles
`CudaRunner.cu` with nvcc, and the seam's `VectorDb.CudaCrossCheck` test runs
the SAME query on the GPU and the CPU and asserts the rankings match (scores
within FPC accumulation tolerance). On a GPU-less box the test prints
`SKIP CUDA cross-check (no device)` and still passes.

### CrabsTK Data corpus I/O (optional)

By default the core builds standalone. To enable the `LoadCorpus`/`SaveCorpus`
`TRecord`/`TReader`/`TWriter` path (shares the iGeek RL-gym on-disk format):

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSUBSECONDBD_USE_CRABSTK_DATA=ON \
      -DCRABSTK_DIR=../../CrabsTK
```

The corpus stores each vector's SubsecondId in the `TRecord`'s
`episode_ids[i]` field (the only per-row `IUD` field; `run_id`/`sequence` are
per-batch). Vector rows live in `obs[]` (`ObsLength == Dim`).

## Verification done (this VM, no GPU)

- **CPU seam green**: `subseconddb-gpu` → `PASS 68/68 PASS 25/25`, exit 0.
  Scheduler 68 checks + VectorDb 25 checks (exact top-k both metrics, FIFO
  keys, bounds/capacity, facade CPU end-to-end, GPU-downgrade-to-CPU).
- **CUDA algorithm cross-check green**: `01.VectorDbCrossCheck.cpp` (the
  2-stage per-block-top-K + merge logic the `__global__` kernels encode,
  mirrored host-side) produces the **identical ranking** to the CPU reference
  for both L2sq and InnerProduct over 3000×16 vectors, K=10.
- **CrabsTK corpus round-trip green**: `SaveCorpus` → disk → `LoadCorpus` →
  search, through the real CrabsTK `TRecord`/CRC32/`TReader`/`TWriter`, nearest
  neighbor returns the correct SubsecondId.

## NOT verified on this VM (no GPU driver, no nvcc)

- The actual `cudaMalloc`/`cudaMemcpy`/`<<<>>>` launch in `CudaRunner.cu`
  (needs the box). The host-side **algorithm** is cross-checked; the CUDA
  **memory management + kernel launch** is not, because there is no GPU here.
  Per the standing note, the 580 driver is NOT installed on this VM until the
  cards are passed through.
- Real iGeek embedding vectors (the corpus tests use synthetic unit-basis and
  LCG data; shape is `Capacity=8, Dim=4` — swap in the real
  `VectorDb<1<<20, 384>` for production).

## Production sizing

Typical iGeek embedding index: `VectorDb<1<<20, 384>` (1M vectors × 384-dim)
or `VectorDb<1<<20, 768>`. That's ~1.5 GB (384-dim) or ~3 GB (768-dim) of
VRAM — fits the 16 GB cards with room for the index + working set. The
scheduler's cold-boot budget (PCIe Gen4 ~25 GB/s + 150 ms overhead) predicts
the RAM→VRAM load time; a 3 GB index is ~270 ms, well under the 1000 ms
target.

## Known limitations (not faked)

- Search is **exact** brute-force top-k — correct, but O(N·Dim) per query.
  IVF / HNSW / product-quantization are the later speedups; not implemented
  yet.
- `CudaRunner::UploadVectors` does a **full re-upload** when the index grows
  (append-only, so a full re-upload is correct). A delta-upload is a later
  optimization.
- The CUDA top-k uses a per-block shared-memory bounded heap with a staging
  barrier (O(count/STAGE) syncs, not O(count)). Correct and reasonably fast;
  a warp-level reduction would be faster. Not faked, not optimized yet.

## Copyright

AStarship <https://astarship.net>.
