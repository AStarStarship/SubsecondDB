// Copyright AStarship <https://astarship.net>.
// POSIX <unistd.h> for write() — not a C++ std library header.
#include <unistd.h>
#include <_Config.h>
#include "00.Core.hxx"

using namespace _;

ISN main(ISN arg_count, CHA** args) {
  const CHA* result = SubsecondDbGpuSeamRun(nullptr);
  ISN len = 0;
  while (result[len]) ++len;
  if (write(1, result, (unsigned long)len) < 0) { /* exit below */ }
  if (write(1, "\n", 1) < 0) { /* exit below */ }
  (void)arg_count; (void)args;
  return (result[0] == 'P') ? 0 : 1;
}
