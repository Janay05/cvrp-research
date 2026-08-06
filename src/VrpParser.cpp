#include "VrpParser.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <unordered_map>

namespace {
    // Splits a header line into tokens and returns the value following a KEY[:] marker,
    // e.g. "DIMENSION : 1000000" or "DIMENSION :\t1000000\t" or "DIMENSION: 1000000".
    bool extract_header_value(const std::string& line, const std::string& key, std::string& out) {
        std::istringstream iss(line);
        std::string tok;
        if (!(iss >> tok)) return false;
        if (tok != key && tok != (key + ":")) return false;
        if (tok == key) {
            // next token might be ":" on its own, or the value directly
            if (!(iss >> tok)) return false;
            if (tok == ":") {
                if (!(iss >> tok)) return false;
            }
        } else {
            if (!(iss >> tok)) return false;
        }
        out = tok;
        return true;
    }

    std::string first_token(const std::string& line) {
        std::istringstream iss(line);
        std::string tok;
        iss >> tok;
        return tok;
    }
}

bool load_vrp_file(const std::string& path, Instance& inst) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Failed to open VRP file: " << path << std::endl;
        return false;
    }

    int dimension = -1;
    int64_t capacity = -1;
    std::string line;

    while (std::getline(file, line)) {
        std::string ft = first_token(line);
        if (ft == "NODE_COORD_SECTION") break;
        std::string val;
        if (extract_header_value(line, "DIMENSION", val)) {
            dimension = std::stoi(val);
        } else if (extract_header_value(line, "CAPACITY", val)) {
            capacity = (int64_t)std::stod(val);
        }
        if (ft == "EOF") {
            std::cerr << "VRP file ended before NODE_COORD_SECTION: " << path << std::endl;
            return false;
        }
    }

    if (dimension <= 1) {
        std::cerr << "VRP file missing/invalid DIMENSION: " << path << std::endl;
        return false;
    }
    if (capacity <= 0) {
        std::cerr << "VRP file missing/invalid CAPACITY: " << path << std::endl;
        return false;
    }

    // Raw arrays indexed by file node id (1..dimension).
    std::vector<double> rawX(dimension + 1), rawY(dimension + 1);
    std::vector<int64_t> rawDemand(dimension + 1, 0);
    std::vector<bool> haveCoord(dimension + 1, false);

    for (int read = 0; read < dimension; ++read) {
        int id;
        double x, y;
        if (!(file >> id >> x >> y)) {
            std::cerr << "VRP file: failed reading NODE_COORD_SECTION entry " << read << " of " << path << std::endl;
            return false;
        }
        if (id < 1 || id > dimension) {
            std::cerr << "VRP file: node id " << id << " out of range [1," << dimension << "] in " << path << std::endl;
            return false;
        }
        rawX[id] = x;
        rawY[id] = y;
        haveCoord[id] = true;
    }
    std::getline(file, line); // consume rest of last coord line

    while (std::getline(file, line)) {
        if (first_token(line) == "DEMAND_SECTION") break;
    }

    for (int read = 0; read < dimension; ++read) {
        int id;
        double d;
        if (!(file >> id >> d)) {
            std::cerr << "VRP file: failed reading DEMAND_SECTION entry " << read << " of " << path << std::endl;
            return false;
        }
        if (id < 1 || id > dimension) {
            std::cerr << "VRP file: demand node id " << id << " out of range in " << path << std::endl;
            return false;
        }
        rawDemand[id] = (int64_t)std::llround(d);
    }
    std::getline(file, line); // consume rest of last demand line

    while (std::getline(file, line)) {
        if (first_token(line) == "DEPOT_SECTION") break;
    }

    int depotId = -1;
    while (file >> depotId) {
        if (depotId == -1) break;
        if (depotId < 1 || depotId > dimension) {
            std::cerr << "VRP file: depot id " << depotId << " out of range in " << path << std::endl;
            return false;
        }
        break; // only single-depot instances are supported; take the first id listed
    }
    if (depotId < 1) {
        std::cerr << "VRP file: DEPOT_SECTION missing/empty in " << path << std::endl;
        return false;
    }

    for (int id = 1; id <= dimension; ++id) {
        if (!haveCoord[id]) {
            std::cerr << "VRP file: node " << id << " never appeared in NODE_COORD_SECTION in " << path << std::endl;
            return false;
        }
    }

    inst.n = dimension - 1;
    inst.Q = capacity;
    inst.x.assign(inst.n + 1, 0);
    inst.y.assign(inst.n + 1, 0);
    inst.demand.assign(inst.n + 1, 0);

    inst.x[0] = (Coord)std::llround(rawX[depotId]);
    inst.y[0] = (Coord)std::llround(rawY[depotId]);
    inst.demand[0] = 0;

    int nextId = 1;
    for (int id = 1; id <= dimension; ++id) {
        if (id == depotId) continue;
        inst.x[nextId] = (Coord)std::llround(rawX[id]);
        inst.y[nextId] = (Coord)std::llround(rawY[id]);
        inst.demand[nextId] = rawDemand[id];
        nextId++;
    }

    std::cout << "Loaded VRP file: " << path << " (n=" << inst.n << ", Q=" << inst.Q << ", depot file-id=" << depotId << ")" << std::endl;
    return true;
}
