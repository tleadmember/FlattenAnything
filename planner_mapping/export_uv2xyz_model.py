import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from util.model import FlattenAnythingModel

# Use: python planner_mapping/export_uv2xyz_model.py --ckpt exported/utah_teapot_reso9/fam.pth --out exported/utah_teapot_reso9/fam_uv_mapper.ts


class FamUvMapper(nn.Module):
    """
    TorchScript-exportable wrapper:
    input  uv01: [N,2] in [0,1]
    output xyz:  [N,3]
           n_in: [N,3] inward normals (center-based flip)
           q:    [N,2] deformed UV
           qcyc: [N,2] cycle UV from unwrap(wrap(q))
    """
    def __init__(self, fam: FlattenAnythingModel, center_xyz):
        super().__init__()
        self.fam = fam
        self.register_buffer("center", torch.tensor(center_xyz, dtype=torch.float32).view(1, 3))

    def forward(self, uv_in: torch.Tensor):
        # 1) 
        # g = uv_in * 2.0 - 1.0                 # [N,2]
        g = uv_in                             # [N,2]
        g = g.unsqueeze(0)                   # [1,N,2]

        # 2) Learned map: G -> Q_hat -> (xyz, normal)
        q = self.fam.grid_deforming(g)       # [1,N,2]
        xyz, n = self.fam.wrapping(q)        # [1,N,3], [1,N,3]
        n = F.normalize(n, dim=-1)

        # 3) Cycle UV for consistency checking
        _, q_cycle = self.fam.unwrapping(xyz)

        # 4) Inward normal orientation (toward center)
        to_center = F.normalize(self.center.unsqueeze(1) - xyz, dim=-1)
        flip = (n * to_center).sum(dim=-1, keepdim=True) < 0.0
        n_in = torch.where(flip, -n, n)

        return xyz.squeeze(0), n_in.squeeze(0), q.squeeze(0), q_cycle.squeeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to fam.pth")
    ap.add_argument("--out", required=True, help="Output TorchScript file, e.g. fam_uv_mapper.ts")
    ap.add_argument("--center_x", type=float, default=0.0)
    ap.add_argument("--center_y", type=float, default=0.0)
    ap.add_argument("--center_z", type=float, default=0.0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = args.device
    fam = FlattenAnythingModel().to(device)
    fam.load_state_dict(torch.load(args.ckpt, map_location=device))
    fam.eval()

    wrapper = FamUvMapper(
        fam=fam,
        center_xyz=(args.center_x, args.center_y, args.center_z)
    ).to(device).eval()

    # Script and save
    scripted = torch.jit.script(wrapper)
    scripted.save(args.out)
    print(f"Saved TorchScript mapper: {args.out}")


if __name__ == "__main__":
    main()