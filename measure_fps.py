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
from gaussian_renderer import get_ray_directions, get_rays
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel
from gaussian_renderer.fast_renderer import FastRenderer
import time

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    renderer = FastRenderer(views[0], gaussians, pipeline.enable_GLO)

    # warmup
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        camera_inds = {view.uid: i for i, view in enumerate(views)}
        camera_ind = camera_inds[view.uid]
        #view.glo_vector = gaussians.glo[camera_ind]
        if gaussians.glo is not None:
            view.glo_vector = torch.cat(
                [gaussians.glo[camera_ind], torch.tensor([
                        math.log(
                        view.iso * view.exposure / 1000),
                    ], device=gaussians.glo.device)
                 ]
            )
        rendering = renderer.render(view, pipeline, background)

    fps = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # if idx != 424:
        #     continue
        camera_inds = {view.uid: i for i, view in enumerate(views)}
        camera_ind = camera_inds[view.uid]
        #view.glo_vector = gaussians.glo[camera_ind]
        if gaussians.glo is not None:
            view.glo_vector = torch.cat(
                [gaussians.glo[camera_ind], torch.tensor([
                        math.log(
                        view.iso * view.exposure / 1000),
                    ], device=gaussians.glo.device)
                 ]
            )
        st = time.time()
        rendering = renderer.render(view, pipeline, background)
        fps.append(time.time() - st)
        # print(time.time() - st, view.image_width, view.image_height)
        gt = view.original_image[0:3, :, :]
        # torchvision.utils.save_image(frendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    print(1/np.mean(fps))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, checkpoint, opt):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.use_neural_network, dataset.max_opacity)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)
        # if dataset.enable_mip_splatting:
        #     gaussians.enable_mip_splatting(
        #         dataset.low_pass_2d_kernel_size, dataset.low_pass_3d_kernel_size)
        #     gaussians.update_low_pass_filter(scene.getTrainCameras())

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    args.checkpoint = None

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.checkpoint, op.extract(args))
