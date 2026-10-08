// Copyright AStarship <https://astarship.net>.
// POSIX <unistd.h> for write() — not a C++ std library header.
#include <unistd.h>
#include <_Config.h>
#include "00.Core.hxx"
#include "01.VectorDb.hxx"

using namespace _;

namespace {
// Concatenate two null-terminated strings into out (bounded). Returns total
// length written (excluding NUL). No std::string.
ISN Concat(CHA* out, ISN cap, const CHA* a, const CHA* b) {
  ISN n = 0;
  while (*a && n + 1 < cap) out[n++] = *a++;
  if (n + 1 < cap) out[n++] = ' ';
  while (*b && n + 1 < cap) out[n++] = *b++;
  out[n] = 0;
  return n;
}
}  // namespace

ISN main(ISN arg_count, CHA** args) {
  const CHA* sched = SubsecondDbGpuSeamRun(nullptr);
  const CHA* vectordb = SubsecondDbVectorDbSeamRun(nullptr);

  static CHA combined[128];
  Concat(combined, 128, sched, vectordb);

  ISN len = 0;
  while (combined[len]) ++len;
  if (write(1, combined, (unsigned long)len) < 0) { /* exit below */ }
  if (write(1, "\n", 1) < 0) { /* exit below */ }
  (void)arg_count;
  (void)args;
  // Pass only if BOTH seams reported PASS.
  return (sched[0] == 'P' && vectordb[0] == 'P') ? 0 : 1;
}
