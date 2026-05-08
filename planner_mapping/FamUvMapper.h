#pragma once
#include <array>
#include <string>

struct PoseOnSurface {
    std::array<float, 3> xyz{};
    std::array<float, 3> normal_inward{};
    float cycle_err = 0.0f;
    bool valid = false;
};

class FamUvMapper {
public:
    struct Config {
        float cycle_err_max = 0.03f;
        bool enforce_uv_bounds = true; // reject if u/v outside [0,1]
    };

    FamUvMapper(const std::string& torchscript_path, Config cfg = {});
    PoseOnSurface mapUvToSurface(float u, float v) const;

private:
    struct Impl;
    Impl* impl_;   // pimpl to keep header clean
};