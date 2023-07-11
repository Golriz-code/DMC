import torch
from torch import nn
from torch.nn import functional as F
from pointnet2_ops import pointnet2_utils
from extensions.chamfer_dist import ChamferDistanceL1
from .Transformer import PCTransformer
from .build import MODELS
from SAP.src.model import PSR2Mesh
from SAP.src.dpsr import DPSR
import argparse
from SAP.src import utils
from SAP.src.model import Encode2Points
from SAP.src.model import PSR2Mesh
def fps(pc, num,device):
    pc = pc.to(device)
    fps_idx = pointnet2_utils.furthest_point_sample(pc, num) 
    sub_pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return sub_pc


class Fold(nn.Module):
    def __init__(self, in_channel , step , hidden_dim = 512):
        super().__init__()

        self.in_channel = in_channel
        self.step = step
        #self.psr2mesh = PSR2Mesh.apply
        a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).cuda()

        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

    def forward(self, x):
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)

        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)

        return fd2

@MODELS.register_module()
class PoinTr(nn.Module):
    def __init__(self,config, **kwargs):
        super().__init__()
        self.trans_dim = config.trans_dim
        self.knn_layer = config.knn_layer
        self.num_pred = config.num_pred
        self.num_query = config.num_query
        self.model_sap = Encode2Points("SAP/configs/learning_based/noise_small/ours.yaml")
        self.psr2mesh = PSR2Mesh.apply
        self.fold_step = int(pow(self.num_pred//self.num_query, 0.5) + 0.5)
        self.base_model = PCTransformer(in_chans = 3, embed_dim = self.trans_dim, depth = [6, 8], drop_rate = 0., num_query = self.num_query, knn_layer = self.knn_layer)
        self.foldingnet = Fold(self.trans_dim, step = self.fold_step, hidden_dim = 256)  # rebuild a cluster point
        self.dpsr = DPSR(res=(128,128,  128), sig = 2)



        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        self.reduce_map = nn.Linear(self.trans_dim + 1027, self.trans_dim)
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL1()

    def get_loss(self, ret, gt):
        loss_fine = self.loss_func(ret, gt)
        #loss_fine = self.loss_func(ret[1], gt)
        return loss_fine


    def forward(self,xyz,min_gt,max_gt,value_std_pc,value_centroid):


        q, coarse_point_cloud = self.base_model(xyz) # B M C and B M 3
    
        B, M ,C = q.shape

        global_feature = self.increase_dim(q.transpose(1,2)).transpose(1,2) # B M 1024
        global_feature = torch.max(global_feature, dim=1)[0] # B 1024
        #print(global_feature.shape)
        rebuild_feature = torch.cat([
            global_feature.unsqueeze(-2).expand(-1, M, -1),
            q,
            coarse_point_cloud], dim=-1)  # B M 1027 + C

        rebuild_feature = self.reduce_map(rebuild_feature.reshape(B*M, -1)) # BM C
        # # NOTE: try to rebuild pc
        # coarse_point_cloud = self.refine_coarse(rebuild_feature).reshape(B, M, 3)
        #print(rebuild_feature.shape)
        # NOTE: foldingNet
        relative_xyz = self.foldingnet(rebuild_feature).reshape(B, M, 3, -1)    # B M 3 S
        #print(relative_xyz.shape)
        rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-1)).transpose(2,3).reshape(B, -1, 3)  # B N 3
        ##print(rebuild_points.shape)


        # cat the input
        #inp_sparse = fps(xyz, self.num_query)

        # denormalize the data based on mean and std
        value_std_points=value_std_pc.view((rebuild_points.shape[0],1,3))
        value_centroid_points=value_centroid.view((rebuild_points.shape[0],1,3))
        De_point = torch.multiply(rebuild_points, value_std_points) + value_centroid_points

        #Normalize data to min and max on gt
        min_depoint=min_gt.view(rebuild_points.shape[0],1,1)
        max_depoint = max_gt.view(rebuild_points.shape[0],1,1)

        Npoints = torch.div(torch.subtract(De_point,min_depoint),torch.subtract((max_depoint + 1),min_depoint))

        #SAP
        out = self.model_sap(Npoints)
        points, normals = out
        psr_grid= self.dpsr(points, normals)

        #return  psr_grid,points,rebuild_points,min_depoint,max_depoint
        return  psr_grid,rebuild_points

