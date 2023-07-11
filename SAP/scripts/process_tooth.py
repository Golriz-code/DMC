import multiprocessing
import numpy as np
from tqdm import tqdm
from src.dpsr import DPSR
#from easy_mesh_vtk import Easy_Mesh
import pyvista as pv
import os
from pymeshfix._meshfix import PyTMesh
from pymeshfix import MeshFix
import open3d as o3d
import torch

data_path = 'C:\\Users\\Golriz\\OneDrive - polymtl.ca\\Desktop\\datasap'  # path for ShapeNet from ONet
data_path_mesh='C:/Users/Golriz/OneDrive - polymtl.ca/Desktop/mesh'
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
out_path_cur_obj='C:\\Users\\Golriz\\OneDrive - polymtl.ca\\Desktop\\datasap'
dpsr = DPSR(res=(resolution, resolution, resolution), sig=0)





mesh=o3d.io.read_triangle_mesh(os.path.join(data_path_mesh, 'shell26_registered_watertight.ply'))
mesh=mesh.subdivide_loop(number_of_iterations=3)
#mesh = Easy_Mesh(os.path.join(data_path_mesh, '45shell_registered.ply'))
points = mesh.vertices
print('points',np.asarray(points))
Npoints = np.asarray(points)
normals = mesh.compute_vertex_normals()
Nnormals = np.asarray(normals.vertex_normals)
print("normals",np.asarray(normals.vertex_normals))
#randomple sample 100000
points_sample = 100000
positive_mesh_idx = np.arange(len(Npoints))
try:
    positive_selected_mesh_idx = np.random.choice(positive_mesh_idx, size=points_sample, replace=False)
except ValueError:
    positive_selected_mesh_idx = np.random.choice(positive_mesh_idx, size=points_sample, replace=True)
#mesh_with_newpoints = np.zeros([points_sample, Npoints.shape[1]], dtype='float32')
Npoints = Npoints[positive_selected_mesh_idx, :]
print('shape',Npoints.shape)
Nnormals= Nnormals[positive_selected_mesh_idx, :]


# normalize the point to [0, 1)
Npoints=(Npoints-np.min(Npoints))/(np.max(Npoints)+1-np.min(Npoints))

#mean/std
#Npoints_mean=np.mean(Npoints)
#Npoints_std=np.std(Npoints)
#Npoints=(Npoints-Npoints_mean) / Npoints_std
#Npoints = np.asarray(Npoints) / padding + 0.5
# to scale back during inference, we should:
# ! p = (p - 0.5) * padding


if save_pointcloud:
    outdir = os.path.join(out_path_cur_obj, 'pointcloud.npz')
    # np.savez(outdir, points=points, normals=normals)
    np.savez(outdir, points=Npoints, normals=np.asarray(Nnormals))
    # return

if save_psr_field:
    psr_gt = dpsr(torch.from_numpy(Npoints.astype(np.float32))[None],
                  torch.from_numpy(Nnormals[None].astype(np.float32))).squeeze().cpu().numpy().astype(np.float16)

    outdir = os.path.join(out_path_cur_obj, 'psr.npz')
    np.savez(outdir, psr=psr_gt)