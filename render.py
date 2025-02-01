#
# Copyright (C) 2023, Inria, Google
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
import math
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
from gaussian_renderer.ever import splinerender
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel
from scene.dataset_readers import ProjectionType

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # if idx != 424:
        #     continue
        N = 1
        frendering = None
        camera_inds = {view.uid: i for i, view in enumerate(views)}
        for i in range(N):
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
            # view.model=ProjectionType.PERSPECTIVE
            rendering = splinerender(view, gaussians, pipeline, background, random=False)["render"]
            if frendering is None:
                frendering = rendering / N
            else:
                frendering += rendering / N
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(frendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, checkpoint, opt):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.use_neural_network, dataset.max_opacity, dataset.tmin)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

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
    args.checkpoint = args.checkpoint if hasattr(args, "checkpoint") else None

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.checkpoint, op.extract(args))
