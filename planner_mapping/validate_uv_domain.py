import argparse
import numpy as np
import torch
import torch.nn.functional as F

# Use: python planner_mapping/validate_uv_domain.py --ts_model exported/utah_teapot_reso9/fam_uv_mapper.ts --num 20000

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts_model", required=True, help="TorchScript model from export_uv2xyz_model.py")
    ap.add_argument("--num", type=int, default=20000)
    ap.add_argument("--cycle_quantile", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model = torch.jit.load(args.ts_model).eval()

    # Uniform sampling in canonical domain [-1,1]^2
    uv_random = torch.rand(args.num, 2, dtype=torch.float32) * 2.0 - 1.0
    with torch.no_grad():
        xyz, n_in, q, q_cycle = model(uv_random)

    cycle_err = torch.norm(q - q_cycle, dim=-1).cpu().numpy()
    thr = float(np.quantile(cycle_err, args.cycle_quantile))

    print(f"Samples: {args.num}")
    print(f"Cycle error mean: {cycle_err.mean():.6f}")
    print(f"Cycle error p95 : {np.quantile(cycle_err, 0.95):.6f}")
    print(f"Recommended cycle_err_max (@q={args.cycle_quantile}): {thr:.6f}")

    # Save a tiny report for runtime config
    print("\nUse this in config:")
    print(f"cycle_err_max: {thr:.6f}")


if __name__ == "__main__":
    main()