# SubsecondDb Agent Guide

## Kanban board

This project belongs to the **AStarship** organization; its Kanban board slug is `astarship` (see `~/AStarStarship/AGENTS.md`). Always target `astarship` explicitly (`--board astarship` / `board="astarship"`). Never rely on the current-board pointer. If a task's assigned board differs, report the mismatch before acting — do not silently switch.

## Build Commands

- **Build extension**: `cmake -B build && cmake --build build`
- **Run tests**: `ctest --output-on-failure` or `./build/tests/test-name`
- **Format code**: `clang-format -i "**/*.cpp" "**/*.h" "**/*.inl"`
- **Lint**: `clang-tidy -p build **/*.cpp`

## Code Style (Chimera Case + ASCII C++)

### Naming
- **Immutable members**: CamelCase (e.g., `TotalCount`, `MaxSize`)
- **Mutable members**: lower_snake_case (e.g., `total_count`, `max_size`)
- **Functions**: CamelCase (e.g., `CalculateHash`, `ProcessRequest`)
- **Namespaces**: CamelCase (e.g., `Subsecond`, `Auth`, `KVStore`)
- **Macros**: UPPER_SNAKE_CASE

### Formatting
- Use 2-space tabs (no spaces)
- K&R braces: opening brace on same line as control statement
- No indentation for root namespace
- Line width: 100 characters max
- Include order: system headers → Crabs library → PostgreSQL headers → project headers

### Types & Data
- Use ASCII Data Type from the ASCII Data Spec: `CHA*` (char*), ISD: `INT64`, IUD: `UINT32`, FPC: `FLOAT32`, etc.
- **No C++ std library.** Do not include `<string>`, `<vector>`, `<iostream>`,
  `<map>`, `<set>`, `<algorithm>`, or any other C++ standard library header.
  Use ASCII Crabs data types instead. This is a hard prohibition, not a
  preference.
- All contiguous memory ASCII data spec for data-driven design
- Follow Crabs/Crabs API conventions

### Error Handling
- Return error codes as int (0 = success, negative = error)
- Use `SEAM` macros for debug/release branching
- Log errors via seam log mechanism: `seam_log`

## Test Guidelines

- Tests in `Pg/tests/` directory with `_test.cpp` suffix
- Use `TEST_BEGIN`, `TEST_END`, `PRINT_HEADING` macros from `_config.h`
- Run single test: `ctest -R test-name -V`
- All tests must pass before commit

## Chinese Room Architecture

The extension models an office like the Chinese Room thought experiment: the
Postgres process is the "room" that follows the syntax rules (SQL) to produce
correct outputs (query results) without itself "understanding" the content. The
SubsecondId is the office's filing system — every document (row) gets a
time-ordered 64-bit id that locates it by *when* it arrived and *which desk*
(source id) filed it, so the room can retrieve any filing by the id alone, the
way a clerk pulls a file by its number without reading it. The hot tier is the
open desk (in-memory KV), the archive is the records room (Postgres), and the
GPU vector tier is the reference librarian (semantic search over what the
filings say).

## TypeScript/MCP Server

**Not yet present in this repo.** A TypeScript/MCP server for database
configuration is planned (it will live in `Next-Drizzle/`), but that directory
has not been added yet. Do not build against it until it lands. (When it does:
`npm run dev` to run, `npm run build` to build.)

## Git Workflow

- Branch from main for features/fixes
- Commit with concise message: `<type>: <description>`
- Types: feat, fix, docs, style, refactor, test, chore
- Squash merge preferred

## Code Style — Chimera+ (AStarship Standard)

All **code and filenames** in the AStarship ecosystem follow the **Chimera+** style guide
(`~/AStarStarship/ASCIICrabs/__ChimeraPlus.md`). The naming philosophy applies to code
identifiers and filenames fleet-wide.

### Core Rule: immutable vs mutable
- **Immutable** (can't change after init) → **CamelCase**
- **Mutable** (changes at runtime) → **lower_snake_case**

### Naming Summary
| Element | Convention | Example |
|---|---|---|
| Type aliases (fixed-width) | 3-letter CAPS codes | `CHA` char8, `ISC` int32, `IUD` uint64, `BOL` bool, `FPC` float32, `FPD` double64 |
| Structs / classes | CamelCase + T(POD)/A(utoject) prefix | `TArray`, `AMap`, `Crabs` |
| Struct members (mutable) | lower_snake_case | `socket_bytes`, `header_bytes` |
| Struct members (immutable) | CamelCase | `StackTotalMin`, `ColumnWidth` |
| Free functions / methods | CamelCase, type-prefixed | `CrabsInit`, `TArrayBytes`, `TMapFind` |
| Function suffixes | `_NC` (no-check), `C` (const/count), `T` (POD), `A` (autoject) | `TArrayInsert_NC`, `CSizeMin` |
| Local variables | lower_snake_case (always) | `bytes_data`, `read_cursor` |
| Macros / #define | UPPER_SNAKE_CASE | `CRABS_RUN_TESTS`, `CPU_X64` |
| Private/protected members | trailing underscore | `aobj_`, `array_` |
| Namespaces | `namespace _ { ... }` | single shared underscore namespace |
| DB names / tables / columns | lower_snake_case (mutable) | `user_sessions`, `order_items` |
| Client-facing API / headers | CamelCase (immutable contract) | `OrderService`, `PaymentGateway` |
| Filenames (client contract) | CamelCase | `OrderService.h` |
| Filenames (host/storage) | lower_snake_case | `user_sessions.py` |

### Header Boilerplate (C/C++)
```
// Copyright AStarship <https://astarship.net>.
#pragma once
#ifndef CRABS_<NAME>_<H|HPP>
#define CRABS_<NAME>_<H|HPP>
#include "..."
#if SEAM >= CRABS_<NAME>
... body ...
#endif
#endif
```

### Formatting
- 2-space indent
- Opening brace same line (functions/control), next line (structs/classes)
- `D_ASSERT()`, `D_COUT()`, `D_RETURNT(type, val)` for debug/assert/return
- `NILP` = nullptr
- `alignas(ACPUCacheLineSize)` for hot structs
- Doxygen `/** @param @return @pre @link @see @code */` comments

### The One-Sentence Version
**Immutable → CamelCase, mutable → lower_snake_case; macros UPPER_SNAKE; private/protected
get a trailing `_`; types are short CAPS width-codes (CHA/ISC/IUD/BOL); structs are
CamelCase with T(POD)/A(utoject) prefix; functions are CamelCase with T/A kind prefixes
and _NC/C markers; locals are snake_case; everything lives in `namespace _`; every file
starts with AStarship copyright, `#pragma once`, a `CRABS_*` guard, and `#if SEAM >= CRABS_*`
gating.** The name tells you the kind, width, access level, and mutability — before you
read the body.