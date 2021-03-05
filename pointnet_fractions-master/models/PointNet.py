import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from utils.torch_pointcloud_utils import *


class PointNetSetAbstraction(nn.Module):
    """
    Properties:
        net_explore - enable/disable net exploration
    """

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()

        self.npoint = npoint  # number of samples
        self.radius = radius  # local region radius
        self.nsample = nsample  # max number of points in local region

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.group_all = group_all

        self._net_explore = False
        self.points_of_interest = None

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, sampled_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, sampled_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        new_points = sampled_points.permute(0, 3, 2, 1)
        for i, (bn, conv) in enumerate(zip(self.mlp_bns, self.mlp_convs)):
            new_points = F.relu(bn(conv(new_points)))

        max_features = torch.max(new_points, 2)
        new_points = max_features[0]

        # Net exploration - finding points of interests
        if self._net_explore:
            B, N, d = new_xyz.shape

            if B > 1:
                # Problem when trying to create interests_points_pool for batches
                raise RuntimeError("While network is in exploration mode, only one image permitted per batch")

            # Return points to original mapping
            sampled_points_orig = sampled_points[:, :, :, 0:d] + new_xyz.view(B, N, 1, d)

            indices = max_features[1].permute(0, 2, 1)
            mask = torch.zeros(sampled_points_orig.size()[:-1])
            mask.scatter_(2, indices, 1.)

            interests_points_pool = sampled_points_orig[mask.byte(), :]
            interests_points = interests_points_pool[0].unsqueeze(0)
            for point in interests_points_pool:
                diff = torch.sum(torch.abs(interests_points - point), -1)
                if torch.sum(diff <= 0.0001) == 0:
                    interests_points = torch.cat((interests_points, point.unsqueeze(0)), 0)

            self.points_of_interest = interests_points

        return new_xyz.permute(0, 2, 1), new_points

    def set_net_explore(self):
        self._net_explore = True

    def unset_net_explore(self):
        self._net_explore = False

    @property
    def net_explore(self):
        return self._net_explore

class PointNetSetAbstractionMsg(nn.Module): # TODO: Problem initializing super class
    """
    Properties:
        net_explore - enable/disable net exploration
    """
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()

        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list

        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint

        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointsDropout(nn.Module):

    def __init__(self, p=0.95):
        super(PointsDropout, self).__init__()

        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p

    def forward(self, xyz):

        if self.training:
            device = xyz.device
            _, _, npoint = xyz.shape
            theta = np.random.uniform(0,self.p)
            new_npoint = int((1-theta)*npoint)

            indices = torch.randperm(npoint).to(device)[0:new_npoint]
            xyz = xyz[:, :, indices]

        return xyz


class PointNet2(nn.Module):
    """
    Properties:
        pf_dim - (int) set to the number of extra point features you'll input the network 
        net_explore - enable/disable net exploration
    """
    def __init__(self, features_len):
        super(PointNet2, self).__init__()

        self._pf_dim = 0
        self._features_len = features_len
        self._net_explore = False

        self._scale = 'SSG'
        self._nlayers = 3

        self._basename = "PointNet2"
        self._name = "{}_PointFeaturesDims{}_FeaturesLen{}_Scale{}_Layers{}"\
            .format(self._basename, self.pf_dim, self.features_len, self.scale, self._nlayers)

        # self.pdrop = PointsDropout(0.3)

        self.sa = nn.ModuleList()           # npoint, radius, nsample, in_channel, mlp, group_all
        self.sa.append(PointNetSetAbstraction(128, 0.2, 32, 3 + self.pf_dim, [64, 64, 128], False))
        self.sa.append(PointNetSetAbstraction(32, 0.4, 64, 128 + 3, [128, 128, 256], False))
        self.sa.append(PointNetSetAbstraction(None, None, None, 256 + 3, [256, 512, self.features_len], True))

    def forward(self, points):

        # points = self.pdrop(points)

        B, C, _ = points.shape
        assert C == 3 + self.pf_dim, "Dimension error, expected points dimension 3 + %d" % (self.pf_dim,)

        if self.pf_dim == 0:
            l_xyz, l_points = self.sa[0](points[:, 0:3, :], None)
        else:
            l_xyz, l_points = self.sa[0](points[:, 0:3, :], points[:, 3:, :])

        for sa in self.sa[1:]:
            l_xyz, l_points = sa(l_xyz, l_points)

        x = l_points.view(B, self.features_len)

        return x

    def set_network_formation(self, scale='SSG', nlayers=0):

        if scale.upper() == 'MSG':
            print("Change to MSG!")
            input()
            self._scale = 'MSG'
            self._nlayers = 3

            self.sa = nn.ModuleList()
            self.sa.append(PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], 3 + self.pf_dim,
                                                     [[32, 32, 64], [64, 64, 128], [64, 96, 128]]))
            self.sa.append(PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 3 + 320,
                                                     [[64, 64, 128], [128, 128, 256], [128, 128, 256]]))
            self.sa.append(PointNetSetAbstraction(None, None, None, 3 + 640, [256, 512, self.features_len], True))

        else:   # Default scale 'SSG'
            self._scale = 'SSG'

            if nlayers == 4:
                self._nlayers = 4
                self.sa = nn.ModuleList()
                self.sa.append(PointNetSetAbstraction(512, 0.2, 32, 3 + self.pf_dim, [64, 64, 128], False))
                self.sa.append(PointNetSetAbstraction(256, 0.4, 32, 128 + 3, [128,128,256], False))
                self.sa.append(PointNetSetAbstraction(128, 0.6, 32, 256 + 3, [256,256,512], False))
                self.sa.append(PointNetSetAbstraction(None, None, None, 512 + 3, [512, 512, self.features_len], True))
            else:   # Default formation 3 layers
                self._nlayers = 3
                self.sa = nn.ModuleList()
                self.sa.append(PointNetSetAbstraction(512, 0.2, 32, 3 + self.pf_dim, [64, 64, 128], False))
                self.sa.append(PointNetSetAbstraction(128, 0.4, 63, 128 + 3, [128, 128, 256], False))
                self.sa.append(PointNetSetAbstraction(None, None, None, 256 + 3, [256, 512, self.features_len], True))

        print("Changing {} to scale {} and number of layers {}".format(self._basename, self.scale, self.nlayers))
        self._name = "{}_PointFeaturesDims{}_FeaturesLen{}_Scale{}_Layers{}"\
            .format(self._basename, self.pf_dim, self.features_len, self.scale, self._nlayers)

    def export_network(self):
        # print("Exporting PointNet {}".format(self.name))
        pointnet_dict = {}
        pointnet_dict["name"] = self.name
        pointnet_dict["_basename"] = self._basename
        pointnet_dict["pf_dim"] = self.pf_dim
        pointnet_dict["features_length"] = self.features_len
        pointnet_dict["scale"] = self.scale
        pointnet_dict["nlayers"] = self.nlayers
        pointnet_dict["state_dict"] = copy.deepcopy(self.state_dict())
        return pointnet_dict

    def load_network(self, pointnet_dict):
        print("Loading PointNet {}".format(self.name))
        self._basename = pointnet_dict["_basename"]
        self.pf_dim = pointnet_dict["pf_dim"]
        self.features_len = pointnet_dict["features_length"]
        self.set_network_formation(pointnet_dict["scale"], pointnet_dict["nlayers"])
        self.load_state_dict(pointnet_dict["state_dict"])

    def set_net_explore(self):
        self.net_explore = True

    def unset_net_explore(self):
        self.net_explore = False

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise ValueError("name should be a string")
        self._basename = value
        self._name = "{}_PointFeaturesDims{}_FeaturesLen{}_Scale{}_Layers{}" \
            .format(self._basename, self.pf_dim, self.features_len, self.scale, self._nlayers)

    @property
    def net_explore(self):
        return self._net_explore

    @net_explore.setter
    def net_explore(self, value):
        if not isinstance(value, bool):
            raise ValueError("net_explore should be 'bool'")
        self._net_explore = value
        print("{}: Set net_explore {}".format(self.name, value))
        for sa in self.sa:
            sa.net_explore = value

    @property
    def pf_dim(self):
        return self._pf_dim

    @pf_dim.setter
    def pf_dim(self, value):
        self._pf_dim = value
        self._name = "{}_PointFeaturesDims{}_FeaturesLen{}_Scale{}_Layers{}" \
            .format(self._basename, self.pf_dim, self.features_len, self.scale, self._nlayers)
        self.set_network_formation(self.scale, self.nlayers)
        print("{}: Set points features dim to {}".format(self.name, value))

    @property
    def features_len(self):
        return self._features_len

    @features_len.setter
    def features_len(self, value):
        self._features_len = value
        self._name = "{}_PointFeaturesDims{}_FeaturesLen{}_Scale{}_Layers{}" \
            .format(self._basename, self.pf_dim, self.features_len, self.scale, self._nlayers)
        self.set_network_formation(self.scale, self.nlayers)
        print("{}: Set output features vector length to {}".format(self.name, value))

    @property
    def scale(self):
        return self._scale

    @property
    def nlayers(self):
        return self._nlayers
