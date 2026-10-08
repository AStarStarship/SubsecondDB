// Copyright AStarship <https://astarship.net>.
#pragma once
#ifndef PACKAGE_CONFIGURATION
#define PACKAGE_CONFIGURATION 1
#include "../../ASCIICrabs/_ConfigHeader.h"

// Use a real ASCIICrabs seam number so the _ConfigFooter.h / _Seams.h
// chain is satisfied. We don't compile ASCIICrabs' own test seams — only
// its type headers — so the seam value just needs to be valid.
#define SEAM CRABS_RELEASE

#include "../../ASCIICrabs/_ConfigDefault.h"
#include "../../ASCIICrabs/_ConfigFooter.h"
#endif
