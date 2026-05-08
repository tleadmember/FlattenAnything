#include "FamUvMapper.h"
#include <torch/script.h>
#include <memory>
#include <stdexcept>

struct FamUvMapper::Impl {
    torch::jit::script::Module mod;
    Config cfg;
};

FamUvMapper::FamUvMapper(const std::string& torchscript_path, Config cfg) {
    impl_ = new Impl();
    impl_->mod = torch::jit::load(torchscript_path);
    impl_->mod.eval();
    impl_->cfg = cfg;
}

PoseOnSurface FamUvMapper::mapUvToSurface(float u, float v) const {
    PoseOnSurface out{};

    if (impl_->cfg.enforce_uv_bounds) {
        if (u < -1.0f || u > 1.0f || v < -1.0f || v > 1.0f) return out;
    }

    // Input [1,2]
    torch::Tensor uv = torch::tensor({{u, v}}, torch::kFloat32);

    // Forward returns tuple: xyz[N,3], n_in[N,3], q[N,2], q_cycle[N,2]
    auto result = impl_->mod.forward({uv}).toTuple();
    torch::Tensor xyz = result->elements()[0].toTensor();
    torch::Tensor n_in = result->elements()[1].toTensor();
    torch::Tensor q = result->elements()[2].toTensor();
    torch::Tensor q_cycle = result->elements()[3].toTensor();

    torch::Tensor cerr = torch::norm(q - q_cycle, 2, -1); // [N]
    float cycle_err = cerr[0].item<float>();
    out.cycle_err = cycle_err;

    if (cycle_err > impl_->cfg.cycle_err_max) return out;

    auto p = xyz[0];
    auto n = n_in[0];

    out.xyz = {p[0].item<float>(), p[1].item<float>(), p[2].item<float>()};
    out.normal_inward = {n[0].item<float>(), n[1].item<float>(), n[2].item<float>()};
    out.valid = true;
    return out;
}