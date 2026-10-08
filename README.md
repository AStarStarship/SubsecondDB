# SubsecondDb

**SubsecondDb is a contiguous, hot-to-archive database for real-time content.**
One 64-bit [SubsecondId](https://github.com/AStarStarship/SubsecondId) addresses
data across three tiers — an in-memory **hot KV** (one inode lookup, no query),
a **GPU vector DB** (semantic search), and **Postgres** (durable archive) — so
you get the fastest possible lookup on hot data and cheap durable storage on
cold data, with the id as the single key tying the tiers together.

Built on [SubsecondId](https://github.com/AStarStarship/SubsecondId)
(64-bit time-ordered hot UUIDs) and [Crabs](https://github.com/AStarStarship/Crabs)
(ASCII contiguous data types, no C++ std lib). The design is all-contiguous,
data-driven, multi-threaded: the id is a flat bit-field you dereference with one
load, the hot tier is a contiguous in-memory KV, and the vector tier runs on the
GPU.

## Phased roadmap (current scope)

**The ASCII Crabs overhaul freezes the deep SubsecondDb work for now.** We build
against **SubsecondId + plain C** (Postgres extension C, no Crabs types) until
the Crabs overhaul lands, then re-adopt Crabs types. The in-memory **KV store is
compartmentalized away from Postgres** — a separate module, not coupled, not
built yet. Phases:

| Phase | Scope | Status |
|---|---|---|
| **1 — Postgres + SubsecondId** | The `Pg/` extension: SubsecondId as a native 64-bit SQL type, hot-to-archive compaction as a function, the collision ledger as a table. **All we need now is Postgres.** No KV, no auth, no IPC, no CUDA. | **current** — `Pg/` is scaffold; the extension C source is the build target. |
| **2 — Auth container + IPC** | The in-memory micro-auth service (username + password, SubsecondId-keyed) as a container, with interprocess communication between it and the Postgres tier. The KV store compartment lands here (separate from Postgres). | next |
| **3 — CUDA-accelerated vector DB** | The GPU tier. The `Gpu/` scheduler (built + seam-tested, 68/68) becomes the placement brain for the vector index (pgVector + CUDA, V100/5060 Ti). | after |

The three-tier vision above is the **end state**; the phases are the build order.
Today, "SubsecondDb" effectively means **the Postgres extension + SubsecondId**,
with the GPU scheduler already built and the rest compartmentalized for later.

## Why hot-to-archive

Online content has a short hot life (minutes to weeks) but a long archive life
(years). SubsecondDb matches storage to that: the **hot** tier is in-memory KV
+ GPU vector (fast, for the data being served *now*), and the **archive** tier is
Postgres (cheap, durable, for everything else). Data moves hot→archive on a
**3-month cadence** inside the SubsecondId's **4.25-year Hot UUID epoch** — so
anything generated 2 months ago is already cold by the time the window compacts.

The **64-bit Hot UUID** is the hot-tier inode:

```
+--------------------------------------------------------------------+
| MSb:1 | 27-bit seconds | 8-bit subsecond ticker | 28-bit source id |
+--------------------------------------------------------------------+
```

- **seconds** (27-bit) — time-ordered; search by date range *without a DB query*
  (the id's seconds field IS the time).
- **ticker** (8-bit) — intra-second uniqueness per source (256 ids/sec/source).
- **source id** (28-bit) — set once at boot per machine/thread; the shard key
  *and* the forensic anchor (which node wrote this row).

The other 64 bits of the full 128-bit identity live in the SQL row (a column) as
the exact-identity / collision tiebreaker. Collisions are extremely infrequent
(the source field disambiguates machines, the ticker disambiguates intra-second);
the rare hot→archive collision (where the Cold pattern drops the source field) is
handled by a small **collision ledger** on the cold table — 8-bit `(before, after)`
counts, almost all zero, that let you scan a collision cluster in O(1).

## The three tiers

| Tier | What | Why |
|---|---|---|
| **Hot KV** (in-memory) | contiguous KV, key = 64-bit Hot UUID | one inode lookup, sub-ms, no query. The primary task. |
| **Vector DB** (GPU, [pgVector](https://github.com/pgvector/pgvector) + CUDA) | embeddings + similarity search | semantic search over content. CPU-only OR GPU-accelerated (runtime mode, not a hardware lock). |
| **Archive** (Postgres + [Redis](https://redis.io) cache) | durable rows + the collision ledger | cheap, durable, time-ranged by the id. |

The two are bridged by the **SCRIPT Protocol** (the AStar scripting/automation
layer) and the **Automaton SCII Crabs** state machine (contiguous stack data,
pop it agentically to undo, A\* over the tree). Auth is an in-memory micro-auth
service keyed by the same SubsecondId (username + password, no email — email is
doxing).

## Hardware: the two-box fleet

Dev and prod are identical boxes:

```
Xeon Gold 6230 (96 GB RAM @ 130 GB/s)
+ 1x V100 16 GB          (FP64, quantum, vector-search)
+ 1x RTX 5060 Ti 16 GB   (H.265 10-bit 4:2:2, Isaac Sim, general)
```

- **Dev box** — runs the CUDA jobs + the vector DB during development.
- **Prod box** — the vector DB runs GPU-accelerated here (the index lives in
  VRAM for fast search); the V100 carries the FP64/quantum + vector-search
  workload, the 5060 Ti carries encode/sim/general.

CPUs are the fallback: a node with no usable GPU runs the vector DB in **CPU
mode** (RAM-resident index) — the same API, no hardware lock.

## GPU Scheduler

`Gpu/` — backend-agnostic GPU scheduler core (C++23, **no C++ std lib**, ASCII
Crabs types, status codes not exceptions). This is the **scheduler brain** for
the two-box fleet: it knows which accelerator is free, which capabilities a job
needs, and whether the memory fits. It does **not** drive any GPU — that is the
runner's job (CUDA-only *for now*).

**Matching is on capabilities, never on card name or backend:**

| `GpuCap` | Meaning | V100 | 5060 Ti | CPU |
|---|---|---|---|---|
| `CapFp64` | real FP64 throughput | ✔ | — | — |
| `CapVideoEncode` | H.265 10-bit 4:2:2 | — | ✔ | — |
| `CapSim` | Isaac Sim / GL desktop | — | ✔ | — |
| `CapVectorSearch` | hold + serve a vector index | ✔ | ✔ | ✔ (RAM) |
| `CapLargeMem` | ≥ 16 GB addressable | ✔ | ✔ | ✔ (RAM) |

A job (`GpuJob`) requires a subset of caps + a memory budget; placement is the
subset test `(accel.caps & job.required_caps) == job.required_caps` plus "free
memory fits" plus "an active runner exists for that backend" (CPU-mode jobs
never need a runner). The runner backend is a **pluggable tag** (`RunnerBackend`)
— swapping to HIP/LevelZero later is a runner replacement, not a scheduler
rewrite.

**Vector DB cold-boot** (GPU mode only): `VectorDbColdBoot()` returns the
RAM→VRAM transfer budget. The bottleneck is **PCIe Gen4 x16 (~25 GB/s)**, not
DRAM (130 GB/s); the target is **≤ 1000 ms** (`ColdBootTargetMs`). CPU mode has
no cold-boot (the index is already RAM-resident) and returns 0.

**Build & test:**
```bash
cd Gpu
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/subseconddb-gpu   # 14 seam tests, 68 checks
```

**Files:**
- `GpuScheduler.h` — `GpuCap`/`GpuJob`/`Gpu`/`GpuScheduler` + status codes
- `GpuScheduler.hxx` — implementation
- `_Seams/00.Core.hxx` — 14 seam tests · `_Seams/_Main.cpp` — entrypoint ·
  `_Seams/_Config.h` — ASCIICrabs config chain · `CMakeLists.txt` — build

## Postgres extension

`Pg/` — the Postgres extension that exposes the archive tier + the collision
ledger as SQL types/operators (SubsecondId as a 64-bit type, hot-to-archive
compaction as a function, the collision ledger as a table). **Status: scaffold**
— the build files + seam harness are in place; the extension C source lands next.
The GPU scheduler above is the first fully-built, seam-tested component.

## Quickstart

```bash
# 1. Clone the stack
git clone https://github.com/AStarStarship/Crabs
git clone https://github.com/AStarStarship/SubsecondId
git clone https://github.com/AStarStarship/SubsecondDb

# 2. Build + test the GPU scheduler (the built component)
cd SubsecondDb/Gpu
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/subseconddb-gpu        # 14 seam tests, 68 checks

# 3. (Postgres extension — lands next)
#    cd SubsecondDb/Pg && cmake -B build && cmake --build build && ctest
```

## Repository layout

```
SubsecondDb/
  Gpu/            GPU scheduler core (built, seam-tested)
  Pg/             Postgres extension (scaffold)
  _impl.hxx       impl include chain
  _seams.hxx      seam include chain
  AGENTS.md       agent guide (build, Chimera+ style, test conventions)
```

## License

Copyright AStarship <https://astarship.net>. Postgres Licensed:

Permission to use, copy, modify, and distribute this software and its
documentation for any purpose, without fee, and without a written agreement is
hereby granted, provided that the above copyright notice and this paragraph and
the following two paragraphs appear in all copies.

IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR DIRECT,
INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE
UNIVERSITY OF CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING,
BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND
THE UNIVERSITY OF CALIFORNIA HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT,
UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
