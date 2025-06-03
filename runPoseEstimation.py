# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Adapted for Morobot Dataset created by Simon Kulovits
'''
Imports
'''
from estimater import *     # Custom module: contains FoundationPose, ScorePredictor, etc.
from datareader import *    # Custom module: loads RGB-D images, masks, etc. adapted by me
import argparse             # For parsing command-line arguments
import os                   # For filesystem path management
import shutil               # To delete and recreate directories

'''
Main Execution Block
'''
if __name__ == '__main__':
    # Setup argument parser
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))  # Get path to current script

    # List of mesh files to be used for object pose estimation
    mesh_files = [
        f'{code_dir}/demo_data/morobots/mesh/1Ag.obj',
        f'{code_dir}/demo_data/morobots/mesh/1Ay_1.obj',
        f'{code_dir}/demo_data/morobots/mesh/1Ay_2.obj',
        f'{code_dir}/demo_data/morobots/mesh/1By_1.obj',
        f'{code_dir}/demo_data/morobots/mesh/1By_2.obj',
        f'{code_dir}/demo_data/morobots/mesh/3Bg.obj',
    ]

    # Arguments parsing with defaults
    parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/morobots')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--img', type=int, default=0)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    parser.add_argument('--output_dir', type=str, default=f'{code_dir}/output')
    args = parser.parse_args()

    # Logging and seed setup
    set_logging_format()
    set_seed(0)

    # Extract arguments
    debug = args.debug
    debug_dir = args.debug_dir
    output_dir = args.output_dir

    # Clean and create output directories
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(os.path.join(output_dir, 'track_vis'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'ob_in_cam'), exist_ok=True)

    # Initialize model components
    scorer = ScorePredictor()            # Object pose scoring model
    refiner = PoseRefinePredictor()      # Refines coarse poses
    glctx = dr.RasterizeCudaContext()    # GPU rasterizer context

    # Initialize pose estimators and metadata
    estimators = []      # List of FoundationPose instances
    meshes = []          # Mesh objects
    mesh_names = []      # Mesh base names
    bboxes = []          # Bounding boxes
    to_origins = []      # Transformation to mesh-centered coordinates

    # Process each mesh
    for mesh_path in mesh_files:
        mesh = trimesh.load(mesh_path)  # Load 3D model
        meshes.append(mesh)
        mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
        mesh_names.append(mesh_name)

        # Compute bounding box and transform to center mesh
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        to_origins.append(to_origin)
        bboxes.append(np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3))

        # Initialize pose estimator for this mesh
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

    # Load RGB-D scene
    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

    # Start image frame processing
    i = args.img  # Chosen frame index
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    vis = color.copy()  # Image for visualizing results

    # Loop over each object estimator
    for obj_id, est in enumerate(estimators):
        mesh_name = mesh_names[obj_id]
        mask = reader.get_mask(i, mesh_name).astype(bool)  # Get preapred object segmentation mask

        logging.info(f'Estimating pose for object {obj_id}')
        
        # Pose estimation using RGB, depth, and mask
        pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

        # Save transformation matrix (object in camera frame)
        os.makedirs(f'{output_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(f'{output_dir}/ob_in_cam/{reader.id_strs[i]}_obj{obj_id}.txt', pose.reshape(4, 4))

        '''
        Debug Mewnu
        '''
        if debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origins[obj_id])  # Move object to centered coordinates

            # Draw 3D box and XYZ axis
            vis = draw_posed_3d_box(reader.K, img=vis, ob_in_cam=center_pose, bbox=bboxes[obj_id])
            vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.05, K=reader.K, thickness=2, transparency=0, is_input_rgb=True)

            # Save overlay image
            output_path = f'{output_dir}/track_vis/{reader.id_strs[i]}_obj{obj_id}_vis.png'
            cv2.imwrite(output_path, vis[..., ::-1])  # saving image while converting RGB to BGR for OpenCV
        
        if debug >= 2:
            # Save alternative image using imageio
            imageio.imwrite(f'{output_dir}/track_vis/{reader.id_strs[i]}_obj{obj_id}.png', vis)

        if debug >= 3:
            # Save transformed mesh
            m = meshes[obj_id].copy()
            m.apply_transform(pose)
            m.export(f'{output_dir}/model_tf_obj{obj_id}.obj')

            # Save full 3D point cloud of scene
            xyz_map = depth2xyzmap(depth, reader.K)
            valid = depth >= 0.001
            pcd = toOpen3dCloud(xyz_map[valid], color[valid])
            o3d.io.write_point_cloud(f'{output_dir}/scene_complete_obj{obj_id}.ply', pcd)

    print('\a') # Beep sound when done
