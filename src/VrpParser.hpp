#pragma once
#include "Types.hpp"
#include <string>

// Loads a standard CVRPLIB-format .vrp file (NAME/DIMENSION/CAPACITY/EDGE_WEIGHT_TYPE,
// NODE_COORD_SECTION, DEMAND_SECTION, DEPOT_SECTION). Tolerant of the header ": "/":\t"
// spacing variance seen across different instance sources (our own generated files vs.
// the Accorsi & Vigo I-dataset vs. classic CMT/Golden instances). Remaps whatever file
// node id is listed as the depot to internal index 0; all other nodes become 1..n in
// the order they appear in NODE_COORD_SECTION. Returns false (with a message on stderr)
// on any parse failure.
bool load_vrp_file(const std::string& path, Instance& inst);
