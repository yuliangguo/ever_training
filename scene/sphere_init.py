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
from scene import contractions
import torch
import numpy as np
from icecream import ic

def l2_normalize_th(x, eps=torch.finfo(torch.float32).eps):
    """Normalize x to unit length along last axis."""
    return x / torch.sqrt(
        torch.clip(torch.sum(x**2, dim=-1, keepdim=True), eps, None)
    )

def sphere_init(
    center,
    N,
    device,
    min_opacity=0.2,
    opacity_var=0.1,
    scale_multi=3,
    a=20,
    radius=1,
    sh_degree=0,
    scene_scale=1,
    **kwargs
):
    def contraction(x):
        return 1 / (1 - x + 1 / a) - 1 / (1 + 1 / a)

    distance = torch.rand((N, 1), device=device) ** (1 / 3)
    # c = torch.tensor(1.5, device=device)
    c = 2

    # here is the issue. We want a nice constant density field no matter how many d8s there are

    direction = l2_normalize_th(torch.randn((N, 3), dtype=torch.float32, device=device))
    means = c * distance * direction

    area_per_sphere = 4 / 3 * np.pi / N * scale_multi
    side_len = (area_per_sphere / (3 / 4) / np.pi) ** (1 / 3)
    # scales = side_len + side_len/2*(2*torch.rand(N, 3, device=device)-1)
    scales = side_len * torch.ones((N, 3), device=device)
    scales = c * scales * 0.4

    quats = l2_normalize_th(
        2 * torch.tensor(np.random.rand(N, 4), dtype=torch.float32, device=device) - 1
    )

    length = ((scales.detach()).exp() ** 2).sum(dim=-1).sqrt()
    length = 0.1
    desired_opacity = 0.2 + 0.1 * torch.rand((N), device=device)
    calc_density = -torch.log(1 - desired_opacity) / length
    densities = calc_density / 2
    print("Init density max: ", densities.max())

    means, scales, quats, densities = contractions.inv_contract_gaussians_decomposed(
        means, scales, quats, densities
    )

    # length = (scale_activation(scales.detach()) ** 2).sum(dim=-1).sqrt()
    # length = 0.1
    # desired_opacity = 0.2 + 0.1 * torch.rand((N), device=device)
    # calc_density = -torch.log(1 - desired_opacity) / length
    # densities = calc_density / 10

    feats = torch.zeros(
        (N, (sh_degree + 1) ** 2, 3), dtype=torch.float32, device=device
    )
    feats[:, 0:1, :] = torch.tensor(
        np.random.rand(N, 1, 3) * 0.3 + 0.3, dtype=torch.float32, device=device
    )
    # feats = torch.rand((N, (sh_degree+1)**2, 3), dtype=torch.float32, device=device) * 0.3 + 0.3
    return (
        means * scene_scale * radius + center.reshape(1, 3),
        scales * scene_scale * radius,
        quats,
        densities / scene_scale,
        feats,
    )

