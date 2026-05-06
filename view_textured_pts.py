import argparse
import open3d as o3d

# Use: python view_textured_pts.py ./exported/test_model/textured_points_tested_on_mesh_verts.ply

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ply_path", type=str, help="Path to .ply file")
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.ply_path)
    if pcd.is_empty():
        raise ValueError(f"Failed to load point cloud or file is empty: {args.ply_path}")

    print(pcd)  # shows number of points, etc.
    o3d.visualization.draw_geometries([pcd], window_name=args.ply_path)


if __name__ == "__main__":
    main()