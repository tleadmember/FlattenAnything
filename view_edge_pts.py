import argparse
import open3d as o3d

# Use: python view_edge_pts.py ./exported/test_model/edge_points_tested_on_mesh_verts.ply

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ply_path", type=str, help="Path to edge_points_*.ply")
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.ply_path)
    if pcd.is_empty():
        raise ValueError(f"Empty or unreadable point cloud: {args.ply_path}")

    # If no colors are stored, paint all edge points red for visibility.
    if not pcd.has_colors():
        pcd.paint_uniform_color([1.0, 0.0, 0.0])

    print(pcd)
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Edge Points",
        point_show_normal=False
    )


if __name__ == "__main__":
    main()