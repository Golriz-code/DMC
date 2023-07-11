import os
import open3d as o3d
import torch
import time
import multiprocessing
import numpy as np
from tqdm import tqdm
from src.dpsr import DPSR
#from easy_mesh_vtk import Easy_Mesh
import pyvista as pv
import os
from pymeshfix._meshfix import PyTMesh
from pymeshfix import MeshFix

#data_path = 'C:\\Users\\Golriz\\OneDrive - polymtl.ca\\Desktop\\1a9e1fb2a51ffd065b07a27512172330'  # path for ShapeNet from ONet
data_path_mesh='C:\\Users\\Golriz\\OneDrive - polymtl.ca\\Desktop\\data-watertightshell'
data_pred_points="C:\\Users\\Golriz\\OneDrive - polymtl.ca\\Desktop\\test-sap-watertight-5epochs\\test-pointr\\A0_6014-36\\A0_6014-36.npy"
base = 'data'  # output base directory
dataset_name = 'shapenet_psr'
multiprocess = True
njobs = 8
save_pointcloud = True
save_psr_field = True
resolution = 128
zero_level = 0.0
num_points = 100000
padding = 1.2
out_path_cur_obj='C:\\Users\\Golriz\\OneDrive - polymtl.ca\\Desktop\\test-sap-watertight-5epochs\\test-pointr\\A0_6014-36'


dpsr = DPSR(res=(resolution, resolution, resolution), sig=0)
"""""
gt_path = os.path.join(data_path, 'pointcloud.npz')
data = np.load(gt_path)
points = data['points']
normals = data['normals']

# normalize the point to [0, 1)
points = points / padding + 0.5
# to scale back during inference, we should:
# ! p = (p - 0.5) * padding
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.normals= o3d.utility.Vector3dVector(normals)
pcd.paint_uniform_color([0, 0.45, 0])

o3d.visualization.draw_geometries([pcd], point_show_normal=True)

if save_pointcloud:
    outdir = os.path.join(out_path_cur_obj, 'pointcloud.npz')
    # np.savez(outdir, points=points, normals=normals)
    np.savez(outdir, points=data['points'], normals=data['normals'])
    # return

if save_psr_field:
    psr_gt = dpsr(torch.from_numpy(points.astype(np.float32))[None],
                  torch.from_numpy(normals.astype(np.float32))[None]).squeeze().cpu().numpy().astype(np.float16)

    outdir = os.path.join(out_path_cur_obj, 'psr.npz')
    np.savez(outdir, psr=psr_gt)

"""""
#read the meshgt
    #for entry in os.listdir(data_path):

#Open_mesh = pv.read(os.path.join(data_path_mesh,'shell26_registered.ply'))
#meshfix = MeshFix(Open_mesh)
#holes = meshfix.extract_holes()
#meshfix.repair(verbose=True)
#meshfix.save(os.path.join(data_path_mesh,'shell26_registered_watertight.ply' ))

"""""
for cases in os.listdir(data_path_mesh):
    for i in os.listdir(os.path.join(data_path_mesh,cases)):
        if 'shell' in i:
            print('cases:',cases)
            mesh = o3d.io.read_triangle_mesh(os.path.join(data_path_mesh, cases, i))
            mesh = mesh.subdivide_loop(number_of_iterations=3)
            # mesh = Easy_Mesh(os.path.join(data_path_mesh, '45shell_registered.ply'))
            points = mesh.vertices
            print('points', np.asarray(points))
            Npoints = np.asarray(points)
            normals = mesh.compute_vertex_normals()
            Nnormals = np.asarray(normals.vertex_normals)
            print("normals", np.asarray(normals.vertex_normals))
            # randomple sample 100000
            points_sample = 100000
            positive_mesh_idx = np.arange(len(Npoints))
            try:
                positive_selected_mesh_idx = np.random.choice(positive_mesh_idx, size=points_sample, replace=False)
            except ValueError:
                positive_selected_mesh_idx = np.random.choice(positive_mesh_idx, size=points_sample, replace=True)
            # mesh_with_newpoints = np.zeros([points_sample, Npoints.shape[1]], dtype='float32')
            Npoints = Npoints[positive_selected_mesh_idx, :]
            print('shape', Npoints.shape)
            Nnormals = Nnormals[positive_selected_mesh_idx, :]

            # normalize the point to [0, 1)
            Npoints = (Npoints - np.min(Npoints)) / (np.max(Npoints) + 1 - np.min(Npoints))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(Npoints)
            pcd.normals= o3d.utility.Vector3dVector(Nnormals)
            pcd.paint_uniform_color([0, 0.45, 0])

            o3d.visualization.draw_geometries([pcd], point_show_normal=True)
            # mean/std
            # Npoints_mean=np.mean(Npoints)
            # Npoints_std=np.std(Npoints)
            # Npoints=(Npoints-Npoints_mean) / Npoints_std
            # Npoints = np.asarray(Npoints) / padding + 0.5
            # to scale back during inference, we should:
            # ! p = (p - 0.5) * padding

            if save_pointcloud:
                outdir = os.path.join(out_path_cur_obj, cases, 'pointcloud.npz')
                # np.savez(outdir, points=points, normals=normals)
                np.savez(outdir, points=Npoints, normals=np.asarray(Nnormals))
                # return

            if save_psr_field:
                psr_gt = dpsr(torch.from_numpy(Npoints.astype(np.float32))[None],
                              torch.from_numpy(Nnormals[None].astype(np.float32))).squeeze().cpu().numpy().astype(
                    np.float16)

                outdir = os.path.join(out_path_cur_obj, cases, 'psr.npz')
                np.savez(outdir, psr=psr_gt)

"""""

points=np.load(data_pred_points)
# normalize the point to [0, 1)
Npoints = (points - np.min(points)) / (np.max(points) + 1 - np.min(points))
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(Npoints)
pcd.estimate_normals()
pcd_normals = np.asarray(pcd.normals)
#pcd.normals = o3d.utility.Vector3dVector(Nnormals)
pcd.paint_uniform_color([0, 0.45, 0])

#o3d.visualization.draw_geometries([pcd], point_show_normal=True)


if save_pointcloud:
    outdir = os.path.join(out_path_cur_obj, 'pointcloud.npz')
    # np.savez(outdir, points=points, normals=normals)
    np.savez(outdir, points=Npoints, normals=np.asarray(pcd_normals))
    # return











