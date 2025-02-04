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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from icecream import ic
import random
import math
import cv2
import json
import traceback
from utils.system_utils import searchForMaxIteration
import time
from gaussian_renderer.fast_renderer import FastRenderer

renderFunc = splinerender
# renderFunc = render
from scene.dataset_readers import ProjectionType

def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac

PREVIEW_RES_FACTOR = 1

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def set_glo_vector(viewpoint_cam, gaussians, camera_inds):
    camera_ind = camera_inds[viewpoint_cam.uid]
    viewpoint_cam.glo_vector = torch.cat(
        [gaussians.glo[camera_ind], torch.tensor([
                math.log(
                viewpoint_cam.iso * viewpoint_cam.exposure / 1000),
            ], device=gaussians.glo.device)
         ]
    )

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree, dataset.use_neural_network, dataset.max_opacity)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    else:
        load_iteration = -1
        if load_iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
        else:
            loaded_iter = load_iteration
        print("Loading trained model at iteration {}".format(loaded_iter))
        gaussians.load_ply(os.path.join(dataset.model_path,
                                                       "point_cloud",
                                                       "iteration_" + str(loaded_iter),
                                                       "point_cloud.ply"))
    # gaussians.load_ply("output/a5911cf7-0/point_cloud/iteration_30000/point_cloud.ply")
    # gaussians.load_ply("/home/amai/Downloads/point_cloud.ply")
    # gaussians.load_ply("/home/amai/3DGS/output/20e2f33c-e/point_cloud/iteration_30000/point_cloud.ply", legacy_compat=True)
    # gaussians.load_ply("/home/amai/gaussian-splatting/output/242678df-0/point_cloud/iteration_30000/point_cloud.ply")
    # gaussians.training_setup(opt)
    
    if pipe.enable_GLO:
        metadata_path = os.path.join(dataset.source_path, "metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        first_metadata = metadata[list(metadata.keys())[0]]

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)


    if dataset.enable_mip_splatting:
        viewpoint_stack = scene.getTrainCameras().copy()
        ema_loss_for_log = 0.0
        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
        first_iter += 1
        camera_inds = {view.uid: i for i, view in enumerate(viewpoint_stack)}
        gaussians.initialize_glo(len(viewpoint_stack), dataset.glo_latent_dim)
        train_cameras = scene.getTrainCameras()
        gaussians.enable_mip_splatting(dataset.low_pass_2d_kernel_size, dataset.low_pass_3d_kernel_size)
        gaussians.update_low_pass_filter(train_cameras)

    gaussians.training_setup(opt)
    dssim_growth = (opt.max_lambda_dssim - opt.lambda_dssim) / opt.max_dssim_iteration
    torch.cuda.empty_cache()
    st = time.time()


    renderer = None

    while True:
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                # custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                custom_cam, do_training, _, _, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    if pipe.enable_GLO:
                        custom_cam.glo_vector = torch.cat(
                            [gaussians.glo[0], torch.tensor([
                                    math.log(
                                    float(first_metadata['iso']) * convert_to_float(first_metadata['exposure']) / 1000),
                                ], device=gaussians.glo.device)
                             ]
                        )
                    # custom_cam.model = viewpoint_cam.model
                    # custom_cam.distortion_params = viewpoint_cam.distortion_params
                    # custom_cam.model=ProjectionType.FISHEYE
                    custom_cam.model=ProjectionType.PERSPECTIVE
                    # custom_cam.glo_vector = viewpoint_cam.glo_vector
                    image_width = custom_cam.image_width
                    image_height = custom_cam.image_height
                    custom_cam.image_width = image_width // PREVIEW_RES_FACTOR
                    custom_cam.image_height = image_height // PREVIEW_RES_FACTOR

                    if renderer is None:
                        renderer = FastRenderer(custom_cam, gaussians, pipe.enable_GLO)

                    renderer.set_camera(custom_cam)

                    st = time.time()
                    # net_image = renderer.render(custom_cam, pipe, background)

                    net_image = renderFunc(custom_cam, gaussians, pipe, background, scaling_modifer, random=False, tmin=0)["render"]
                    print(1/(time.time()-st))
                    net_image = (torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
                    net_image = cv2.resize(net_image, (image_width, image_height))
                    # ic(net_image.shape, net_image.dtype)
                    net_image_bytes = memoryview(net_image)
                network_gui.send(net_image_bytes, dataset.source_path)
                torch.cuda.empty_cache()
            except Exception as e:
                print(traceback.format_exc())
                network_gui.conn = None


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    # args.checkpoint_iterations.append(args.iterations)
    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.save_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done

