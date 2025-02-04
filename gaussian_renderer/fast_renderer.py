# coding=utf-8
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import math
import numpy as np
from scene import Scene
import os
import math
from tqdm import tqdm
from os import makedirs
from gaussian_renderer.ever import get_ray_directions, get_rays, camera2rays_full
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel
import time

from utils.sh_utils import eval_sh, RGB2SH, SH2RGB
from splinetracer.splinetracers.fast_ellipsoid_splinetracer import sp
from splinetracer.eval_sh import eval_sh as eval_sh2
from utils.graphics_utils import in_screen_from_ndc, project_points, visible_depth_from_camspace, fov2focal
from scene.dataset_readers import ProjectionType


MAX_ITERS = 200

class FastRenderer:
    def __init__(self, view, pc, enable_GLO):
        self.device = pc.get_xyz.device
        self.enable_GLO = enable_GLO
        w = view.image_width
        h = view.image_height

        fx = 0.5 * w / math.tan(0.5 * view.FoVx)  # original focal length
        fy = 0.5 * h / math.tan(0.5 * view.FoVy)  # original focal length

        directions = get_ray_directions(h, w, [fx, fy], random=False).cuda()  # (h, w, 3)
        self.directions = (directions / torch.norm(directions, dim=-1, keepdim=True))
        self.otx = sp.OptixContext(torch.device("cuda:0"))
        self.prims = sp.Primitives(self.device)
        self.mean = pc.get_xyz.contiguous()
        self.quat = pc.get_rotation.contiguous()
        self.pc = pc

        per_point_2d_filter_scale = torch.zeros(self.pc._xyz.shape[0], device=self.device)
        self.per_point_2d_filter_scale = 1
        self.scales, self.density = pc.get_scale_and_density_for_rendering(self.per_point_2d_filter_scale, 1.0)

        color = self.get_color(view)
        half_attribs = torch.cat([self.mean, self.scales, self.quat], dim=1).half().contiguous()
        self.prims.add_primitives(self.mean, self.scales, self.quat, half_attribs, self.density, color)
        self.gas = sp.GAS(self.otx, self.device, self.prims, True, False, True)
        self.forward = sp.Forward(self.otx, self.device, self.prims, False)

    def set_camera(self, view):
        if view.model != ProjectionType.PERSPECTIVE:
            rays_o, rays_d = camera2rays_full(view, random=False)
            self.directions = rays_d
        else:

            w = view.image_width
            h = view.image_height

            fx = 0.5 * w / math.tan(0.5 * view.FoVx)  # original focal length
            fy = 0.5 * h / math.tan(0.5 * view.FoVy)  # original focal length

            directions = get_ray_directions(h, w, [fx, fy], random=False).cuda()  # (h, w, 3)
            self.directions = (directions / torch.norm(directions, dim=-1, keepdim=True))


    def get_color(self, view):
        shs = self.pc.get_features
        # shs[:, (self.pc.active_sh_degree+1)**2:] = 0
        if self.enable_GLO:
            if view.glo_vector is not None:
                glo_vector = view.glo_vector
            else:
                glo_vector = torch.zeros((1, 64), device='cuda')
            shs = self.pc.glo_network(
                glo_vector.reshape(1, -1), shs.reshape(shs.shape[0], -1)
            ).reshape(shs.shape)

        cam_pos = view.camera_center.to(self.device)
        # wct = view.world_view_transform.cuda().float()
        # T = torch.linalg.inv(wct.T)
        # v = T[:3, 2]
        net_color = eval_sh2(self.pc.get_xyz, shs, cam_pos, self.pc.active_sh_degree)
        # ic(net_color, SH2RGB(features))
        net_color = torch.nn.functional.softplus(net_color, beta=10)
        features = RGB2SH(net_color).reshape(-1, 1, 3)
        return features.contiguous()

    def get_rays(self, view):
        T = torch.linalg.inv(view.world_view_transform.T.cuda())
        rays_o, rays_d = get_rays(
            self.directions,
            T,
        )  # both (h*w, 3)
        rays_o = (rays_o).contiguous()
        return rays_o, rays_d

    def trace_rays(self, rayo, rayd, view, tmin, tmax):
        color = self.get_color(view)
        # prims = sp.Primitives(self.device)
        # half_attribs = torch.cat([self.mean, self.scales, self.quat], dim=1).half().contiguous()
        # prims.add_primitives(self.mean, self.scales, self.quat, half_attribs, self.density, color)
        # self.forward = sp.Forward(self.otx, self.device, prims, False)

        self.prims.set_features(color)
        self.forward.update_model(self.prims)

        out = self.forward.trace_rays(self.gas, rayo, rayd, tmin, tmax, MAX_ITERS, 1000)
        return out

    def render(self,
               view,
               pc,
               bg_color: torch.Tensor,
               tmin=None,
               scaling_modifier=1.0):
        rays_o, rays_d = self.get_rays(view)
        out = self.trace_rays(rays_o, rays_d, view, self.pc.tmin if tmin is None else tmin, 1e7)
        iters = out['saved'].iters
        rendered_image = out['color'][:, :3].T.reshape(3, view.image_height, view.image_width)
        return rendered_image


