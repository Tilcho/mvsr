# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Adapted for Morobot Dataset created by Simon Kulovits

from estimater import *
from datareader import *
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))

    # NEW: hardcoded paths for 6 mesh files
    mesh_files = [
        f'{code_dir}/demo_data/morobots/mesh/obj0.obj',
        f'{code_dir}/demo_data/morobots/mesh/obj1.obj',
        f'{code_dir}/demo_data/morobots/mesh/obj2.obj',
        f'{code_dir}/demo_data/morobots/mesh/obj3.obj',
        f'{code_dir}/demo_data/morobots/mesh/obj4.obj',
        f'{code_dir}/demo_data/morobots/mesh/obj5.obj',
    ]

    parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/morobots')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    debug = args.debug
    debug_dir = args.debug_dir
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    # Create one FoundationPose instance per mesh
    estimators = []
    meshes = []
    bboxes = []
    to_origins = []

    for mesh_path in mesh_files:
        mesh = trimesh.load(mesh_path)
        meshes.append(mesh)
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        to_origins.append(to_origin)
        bboxes.append(np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3))
        est = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            debug_dir=debug_dir,
            debug=debug,
            glctx=glctx
        )
        estimators.append(est)

    logging.info("All estimators initialized.")

    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

    # Process ONLY the first frame
    i = 0
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    mask = reader.get_mask(i).astype(bool)

    for obj_id, est in enumerate(estimators):
        logging.info(f'Estimating pose for object {obj_id}')
        pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

        # Save output pose matrix
        os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}_obj{obj_id}.txt', pose.reshape(4, 4))

        if debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origins[obj_id])
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bboxes[obj_id])
            vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)

            output_path = f'{debug_dir}/track_vis/{reader.id_strs[i]}_obj{obj_id}_vis.png'
            cv2.imwrite(output_path, vis[..., ::-1])

        if debug >= 3:
            m = meshes[obj_id].copy()
            m.apply_transform(pose)
            m.export(f'{debug_dir}/model_tf_obj{obj_id}.obj')
            xyz_map = depth2xyzmap(depth, reader.K)
            valid = depth >= 0.001
            pcd = toOpen3dCloud(xyz_map[valid], color[valid])
            o3d.io.write_point_cloud(f'{debug_dir}/scene_complete_obj{obj_id}.ply', pcd)

        if debug >= 2:
            imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}_obj{obj_id}.png', vis)
