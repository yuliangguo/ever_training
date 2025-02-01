#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud, in_screen_from_ndc, project_points, visible_depth_from_camspace, fov2focal
from utils.general_utils import strip_symmetric, build_scaling_rotation
# import tinycudann as tcnn
from icecream import ic
from pathlib import Path
from scene import sphere_init
from utils import safe_math

MAX_PRIMITIVES = 8_000_000
SCALING_MUL = 1
SCALING2RADIUS = SCALING_MUL*3
USE_MAJOR_AXIS = False
IGNORE_PARAM_LIST = ['glo', "glo_network"]

@torch.jit.script
def inv_opacity(y):
    x = (-(1 - y).clip(min=1e-10).log()).clip(min=0)
    return x

@torch.jit.script
def get_major_axis(scales): 
    # scales = scales.detach()
    # convert scales to max length. Each element corresponds to the length of each segment in xyz

    max_integration_length = scales.max(dim=-1, keepdim=True).values * 2

    return max_integration_length

@torch.jit.script
def get_major_axis_density(opacity, scales):
    scales = scales.detach()

    max_integration_length = get_major_axis(scales)

    densities = inv_opacity(opacity) / max_integration_length

    return densities.reshape(-1)
    

@torch.jit.script
def get_minor_axis_density(opacity, scales):
    scales = scales.detach()
    min_integration_length = scales.min(dim=-1, keepdim=True).values * 2

    densities = inv_opacity(opacity).reshape(min_integration_length.shape) / min_integration_length
    return densities
    
def divide_opacity(opacity, scales):
    density = get_minor_axis_density(opacity, scales).reshape(opacity.shape) / 2
    minor_axis = scales.min(dim=-1).values.reshape(opacity.shape)
    minor_opacity = (1 - (-density * minor_axis).exp()).clip(min=0, max=0.99).reshape(opacity.shape)
    return minor_opacity

def f1(x):
    return torch.expm1(x).log()

def f2(x):
    return x + (1 - x.neg().exp()).log()

def inverse_softplus(x):
    big = x > torch.tensor(torch.finfo(x.dtype).max).log()
    return torch.where(
        big,
        f2(x.masked_fill(~big, 1.)),
        f1(x.masked_fill(big, 1.)),
    )

class GloMLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, output_dim),
        )

    def forward(self, glo_latent, input_x):
        out = self.mlp(glo_latent)
        a, b = torch.split(out, out.shape[-1] // 2, dim=-1)
        return input_x * torch.exp(a.clip(max=3)) + b

class GaussianModel:

    def setup_functions(self, max_opacity):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.max_prim_size = 25.0
        self.min_prim_size = SCALING2RADIUS*1e-4

        self.max_opacity = max_opacity
        self.max_planned_opacity = max_opacity
        self.opacity_activation = lambda x: self.max_opacity*torch.sigmoid(x)
        self.inverse_opacity_activation = lambda y: inverse_sigmoid((y/self.max_opacity).clip(min=1e-3, max=0.999))
        self.scaling_activation = lambda x: (torch.nn.functional.softplus(x) + self.min_prim_size).clip(max=self.max_prim_size)
        self.scaling_inverse_activation = lambda y: inverse_softplus((y-self.min_prim_size).clip(min=1e-4, max=self.max_prim_size))

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.density_activation = lambda x: torch.exp(x).clip(max=1000)
        self.inverse_density_activation = lambda y: torch.log(y.clip(min=1e-10))


        self.feature_activation = lambda x:x
        self.inverse_feature_activation = lambda x:x

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int, use_neural_network: bool = False, max_opacity=0.99, tmin=0.2):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.feat_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.death_mark = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions(max_opacity)
        self.glo = None
        self.tmin = tmin

        self.feat_dim = 3*(self.max_sh_degree+1)**2
        self.glo_network = GloMLP(65, self.feat_dim*2).cuda()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.glo,
            self.glo_network
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale, 
        self.glo,
        self.glo_network) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def get_scale_and_opacity_for_rendering(
            self, per_point_2d_filter_scale=0, scaling_modifier=1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the scale and opacity for rendering."""
        opacity = self.opacity_activation(self._opacity)
        scale = self.scaling_activation(self._scaling) * scaling_modifier

        return (scale, opacity)

    def get_scale_and_density_for_rendering(
            self, per_point_2d_filter_scale=0, scaling_modifier=1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        scaling, opacity = self.get_scale_and_opacity_for_rendering(
            per_point_2d_filter_scale, scaling_modifier)
        if USE_MAJOR_AXIS:
            density = get_major_axis_density(opacity, scaling)
        else:
            density = get_minor_axis_density(opacity, scaling)
        return (scaling, density)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self.feature_activation(self._features_dc)
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    def get_density(self, scaling_modifier=1):
        opacity = self.opacity_activation(self._opacity)
        scaling = self.get_scaling * scaling_modifier
        if USE_MAJOR_AXIS:
            return get_major_axis_density(opacity, scaling)
        else:
            return get_minor_axis_density(opacity, scaling)

    @property
    def get_minor_axis_opacity(self):
        if USE_MAJOR_AXIS:
            density = self.get_density()
            minor_axis = self.get_scaling.min(dim=-1).values.reshape(density.shape)
            return (1 - (-density * minor_axis).exp()).clip(min=1e-4, max=0.99)
        else:
            return self.get_opacity

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, num_additional_pts: int, additional_size_multi: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = self.inverse_feature_activation(RGB2SH(fused_color))

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud.cuda()), 0.0000001)
        scales = 1*torch.sqrt(dist2)[...,None].repeat(1, 3)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")
        rots = torch.nn.functional.normalize(rots)


        # add points using sphere init
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        if num_additional_pts > 0:
            center = torch.mean(fused_point_cloud, dim=0)
            sph_means, sph_scales, sph_quats, sph_densities, sph_feats = \
                sphere_init.sphere_init(center, num_additional_pts, torch.device("cuda"), a=20, radius=2*fused_point_cloud.std(dim=0).max(), scale_multi=additional_size_multi)

            fused_point_cloud = torch.cat([fused_point_cloud, sph_means], dim=0)
            sph_scales = sph_scales.mean(dim=-1, keepdim=True).expand(-1, 3)
            scales = torch.cat([scales, sph_scales], dim=0).clip(min=self.min_prim_size, max=self.max_prim_size)
            rots = torch.cat([rots, sph_quats], dim=0)
            sph_features = self.inverse_feature_activation(RGB2SH(0.5*torch.ones((sph_means.shape[0], 3, 1), device='cuda')))
            features = torch.cat([features, torch.cat([
                sph_features,
                torch.zeros((sph_means.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1), device='cuda')
            ], dim=2)], dim=0)
            dist = torch.linalg.norm(sph_means, dim=-1, keepdim=True)
            scaling = 1-1/(dist+1)
            opacities = torch.cat([opacities, inverse_sigmoid(0.1 * scaling)], dim=0)

        raw_scales = self.scaling_inverse_activation(scales)

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(raw_scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.per_point_3d_filter_scale = torch.zeros(
            [scales.shape[0], 1], dtype=torch.float32, device='cuda'
        )

    def initialize_glo(self, n, glo_latent_dim):
        self.glo = nn.Parameter(torch.zeros((n, glo_latent_dim), device='cuda'))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.feat_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.death_mark = torch.zeros((self.get_xyz.shape[0]), dtype=int, device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_rest_lr, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': self.glo_network.parameters(), 'lr': training_args.glo_network_lr, "name": "glo_network"},
        ]

        if self.glo is not None:
            l.append(
                {'params': [self.glo], 'lr': training_args.glo_lr, "name": "glo"}
            )

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15, betas=[0.9, 0.999])
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        # f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_dc = self._features_dc.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self, value=0.01):
        # set opacity such that the density accumulated along the major axis produces the target opacity
        target_opacity = torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*value)
        if USE_MAJOR_AXIS:
            opacities_new = self.inverse_opacity_activation(target_opacity).reshape(target_opacity.shape)
        else:
            scaling = self.get_scaling
            target_density = get_major_axis_density(target_opacity, scaling)
            # convert target density to the minor axis opacity
            minor_opacity = (1 - (-target_density * scaling.min(dim=-1).values).exp()).clip(min=1e-3, max=0.99)
            opacities_new = self.inverse_opacity_activation(minor_opacity.reshape(target_opacity.shape))
        opacities_old = self._opacity
        # opacities_new = torch.where(torch.rand_like(opacities_new) > 0.5, opacities_new, opacities_old)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, legacy_compat=False):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], (self.max_sh_degree + 1) ** 2 - 1, 3))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        if legacy_compat:
            ic(legacy_compat)
            features_extra = torch.tensor(features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))).permute(0, 2, 1)
            scales = self.scaling_inverse_activation(self.scaling_activation(torch.as_tensor(scales)) * SCALING2RADIUS)

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree
        self.max_opacity = self.max_planned_opacity

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] in IGNORE_PARAM_LIST:
                continue
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if "exp_avg" in stored_state:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] in IGNORE_PARAM_LIST:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                if "exp_avg" in stored_state:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.feat_gradient_accum = self.feat_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.death_mark = self.death_mark[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] in IGNORE_PARAM_LIST:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                if "exp_avg" in stored_state:
                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.feat_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.death_mark = torch.cat([
            self.death_mark, 
            torch.zeros((self.get_xyz.shape[0] - self.death_mark.shape[0]), device='cuda', dtype=int)
        ], dim=0)

    def densify_and_explore(self, scaling, density, grads, grad_threshold, radius, scene_extent, min_opacity=0.025, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()

        padded_radius = torch.zeros((n_init_points), device="cuda")
        padded_radius[:radius.shape[0]] = radius.squeeze()
        multi = 0.7*N

        probing_axis = get_major_axis(scaling).reshape(density.shape) / np.sqrt(2)
        probing_axis = (scaling**2).sum(dim=-1).sqrt().reshape(density.shape)
        major_opacity = (1 - (-density * probing_axis).exp()).clip(min=0).reshape(-1)

        selected_pts_mask = (padded_grad >= grad_threshold) | (major_opacity > 0.99)
        size_mask = torch.min(self.get_scaling, dim=1).values*SCALING2RADIUS > self.percent_dense*scene_extent / multi
        opacity_mask = (self.get_opacity > min_opacity).reshape(-1)
        selected_pts_mask = selected_pts_mask & size_mask & opacity_mask

        #print(
        #    f"Explored {selected_pts_mask.sum()}/{selected_pts_mask.shape[0]} primitives. Size mask: {size_mask.sum()} Grad mean: {padded_grad.mean()}"
        #)

        # print(
        #         f"Explored {selected_pts_mask.sum()}/{selected_pts_mask.shape[0]} primitives. Size mask: {size_mask.sum()} Gradient: {grads.mean()}"
        # )
        #
        stds = scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds/SCALING2RADIUS)
        rots = build_rotation(self.get_rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        div_scaling = self.scaling_inverse_activation(
            (scaling[selected_pts_mask] / multi).clip(min=5e-3))
        new_scaling = div_scaling.repeat(N,1)
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)

        replace_opacity = self.get_opacity
        divided_opacities = divide_opacity(
            replace_opacity[selected_pts_mask], div_scaling).reshape(-1, 1)
        new_opacity = self.inverse_opacity_activation(divided_opacities.repeat(N,1))

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        return prune_filter

    def densify_and_clone(self, scaling, density, grads, grad_threshold, radius, scene_extent, min_clone_opacity=0.0):
        # Extract points that satisfy the gradient condition
        major_axis = get_major_axis(scaling).reshape(density.shape) / np.sqrt(2)
        major_opacity = (1 - (-density * major_axis).exp()).clip(min=0).reshape(-1)

        selected_pts_mask = (grads >= grad_threshold).reshape(-1)# | (major_opacity > 0.99)

        size_mask = torch.min(scaling, dim=1).values <= self.percent_dense*scene_extent/SCALING2RADIUS
        opacity_mask = ((1 - (1-self.get_opacity).sqrt()) > min_clone_opacity).reshape(-1)
        selected_pts_mask = selected_pts_mask & size_mask & opacity_mask

        replace_opacity = self.get_opacity
        divided_opacities = divide_opacity(replace_opacity[selected_pts_mask], scaling[selected_pts_mask]).reshape(-1, 1)
        replace_opacity[selected_pts_mask] = divided_opacities
        raw_opacity = self.inverse_opacity_activation(replace_opacity)

        optimizable_tensors = self.replace_tensor_to_optimizer(raw_opacity, "opacity")
        self._opacity = optimizable_tensors["opacity"]

        new_opacity = self.inverse_opacity_activation(divided_opacities)

        # print(
        #     f"Cloned {selected_pts_mask.sum()}/{selected_pts_mask.shape[0]} primitives. Size mask: {size_mask.sum()}. Max gradient: {grads.max()}"
        # )
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        new_scaling = torch.cat([
            scaling,
            scaling[selected_pts_mask]
        ], dim = 0)
        new_density = torch.cat([
            density,
            density[selected_pts_mask]
        ], dim = 0)
        return new_scaling, new_density

    def decrease_opacity(self, amount):
        opacities_new = self.inverse_opacity_activation((self.get_opacity - amount))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    @torch.no_grad
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, max_var=0, min_split_opacity=0.025):
        min_denom = 1
        denom = self.denom
        xyz_gradient_accum = self.xyz_gradient_accum
        grads = (self.xyz_gradient_accum / self.denom)
        grads[grads.isnan()] = 0.0
        grads = torch.norm(grads, dim=-1).reshape(-1)
        xyz_grad = self._xyz.grad
        scaling, density = self.get_scale_and_opacity_for_rendering()

        prune_mask = self.update_death_mark()
        if prune_mask.sum() > 0:
            grads = grads[~prune_mask]
            denom = denom[~prune_mask]
            scaling = scaling[~prune_mask]
            density = density[~prune_mask]

        prune_mask = (self.get_minor_axis_opacity < 0.005).reshape(-1)
        n = denom.shape[0]
        prune_mask[:n] = (denom.reshape(-1) > 0) & prune_mask[:n]
        prune_mask[n:] = False

        # print(f"Pruned {prune_mask.sum()} primitives. Mean Prune Opacity: {self.get_opacity[prune_mask].mean()}")
        self.prune_points(prune_mask)

        if prune_mask.sum() > 0:
            grads = grads[~prune_mask]
            scaling = scaling[~prune_mask]
            density = density[~prune_mask]

        if self.get_xyz.shape[0] < MAX_PRIMITIVES:
            radius = self.max_radii2D
            scaling, density = self.densify_and_clone(scaling, density, grads, max_var, radius, extent, min_split_opacity)
            prune_filter1 = self.densify_and_explore(scaling, density, grads, max_grad, radius, extent, min_split_opacity)
            self.prune_points(prune_filter1)

        torch.cuda.empty_cache()

    def update_death_mark(self):
        self.death_mark = (self.death_mark + torch.where(self.max_radii2D.reshape(-1) == 0, 1, -1)).clip(min=-5, max=5)
        
        small_points = self.death_mark > 3

        prune_mask = small_points
        # print(f"Pruned {prune_mask.sum()} primitives. Death Mark: {self.death_mark.float().mean()}")
        if prune_mask.sum() > 0:
            self.prune_points(prune_mask)
        return prune_mask

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
