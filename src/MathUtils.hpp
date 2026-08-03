#pragma once
#include <cstdint>
#include <algorithm>

inline uint64_t xy2d(uint32_t order, uint32_t x, uint32_t y) {
    uint64_t d = 0;
    for (uint32_t s = 1u << (order - 1); s > 0; s >>= 1) {
        uint32_t rx = (x & s) > 0;
        uint32_t ry = (y & s) > 0;
        d += (uint64_t)s * s * ((3 * rx) ^ ry);
        // rotate
        if (ry == 0) {
            if (rx == 1) {
                x = s - 1 - x;
                y = s - 1 - y;
            }
            std::swap(x, y);
        }
    }
    return d;
}
