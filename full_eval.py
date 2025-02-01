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

import os
from argparse import ArgumentParser

GLO_SCENES = ["alameda"]

# mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
# mipnerf360_indoor_scenes = ["counter", "kitchen", "bonsai"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"]
zipnerf_scenes = ["alameda", "nyc", "london", "berlin"]


parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval")
args, _ = parser.parse_known_args()

parser.add_argument('--mipnerf360', "-m360", default='', type=str)
parser.add_argument("--tanksandtemples", "-tat", default='', type=str)
parser.add_argument("--deepblending", "-db", default='', type=str)
parser.add_argument("--zipnerf", "-zn", default='', type=str)
parser.add_argument("--skip_360_indoor", action='store_true')
parser.add_argument("--skip_360_outdoor", action='store_true')
parser.add_argument("--port", default=6009, type=int)
parser.add_argument("--additional_args", default="", type=str)
args = parser.parse_args()

if args.skip_360_outdoor:
    mipnerf360_outdoor_scenes = []
if args.skip_360_indoor:
    mipnerf360_indoor_scenes = []
if len(args.mipnerf360) == 0:
    mipnerf360_indoor_scenes = []
    mipnerf360_outdoor_scenes = []
if len(args.tanksandtemples) == 0:
    tanks_and_temples_scenes = []
if len(args.deepblending) == 0:
    deep_blending_scenes = []
if len(args.zipnerf) == 0:
    zipnerf_scenes = []

all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(mipnerf360_indoor_scenes)
all_scenes.extend(tanks_and_temples_scenes)
all_scenes.extend(deep_blending_scenes)
all_scenes.extend(zipnerf_scenes)

if not args.skip_training:
    common_args = f" --quiet --eval --test_iterations -1 --port {args.port} {args.additional_args}"
    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene
        os.system("python train.py -s " + source + " -i images_4 -m " + args.output_path + "/" + scene + common_args)
    for scene in mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene
        os.system("python train.py -s " + source + " -i images_2 -m " + args.output_path + "/" + scene + common_args)
    for scene in tanks_and_temples_scenes:
        source = args.tanksandtemples + "/" + scene
        os.system(
            "python train.py -s "
            + source
            + " -m "
            + args.output_path
            + "/"
            + scene
            + common_args
        )
    for scene in deep_blending_scenes:
        source = args.deepblending + "/" + scene
        os.system(
            "python train.py -s "
            + source
            + " -m "
            + args.output_path
            + "/"
            + scene
            + common_args
        )
    for scene in zipnerf_scenes:
        glo_args = ""
        if scene in GLO_SCENES:
            glo_args = " --enable_GLO --glo_lr 0 --checkpoint_iterations 7000 30000 "
        source = args.zipnerf + "/" + scene
        os.system(
            "python train.py -s "
            + source
            + " -m "
            + args.output_path
            + "/"
            + scene
            + common_args
            + glo_args
            + " -r 1 --images images_2 "
            + " --position_lr_init 4e-5 --position_lr_final 4e-7 "
            + " --percent_dense 0.0005 --tmin 0"
        )

if not args.skip_rendering:
    all_sources = []
    for scene in mipnerf360_outdoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in mipnerf360_indoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in tanks_and_temples_scenes:
        all_sources.append(args.tanksandtemples + "/" + scene)
    for scene in deep_blending_scenes:
        all_sources.append(args.deepblending + "/" + scene)

    common_args = " --quiet --eval --skip_train"
    for scene, source in zip(all_scenes, all_sources):
        # glo_args = ""
        # if scene in GLO_SCENES:
        #     glo_args = f" --checkpoint {os.path.join(args.output_path,scene,'chkpnt7000.pth')} "
        # os.system(
        #     "python render.py --iteration 7000 -s "
        #     + source
        #     + " -m "
        #     + args.output_path
        #     + "/"
        #     + scene
        #     + common_args
        #     + glo_args
        # )
        glo_args = ""
        if scene in GLO_SCENES:
            glo_args = f" --checkpoint {os.path.join(args.output_path,scene,'chkpnt30000.pth')} "
        os.system(
            "python render.py --iteration 30000 -s "
            + source
            + " -m "
            + args.output_path
            + "/"
            + scene
            + common_args
            + glo_args
        )

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += '"' + args.output_path + "/" + scene + '" '

    os.system("python metrics.py -m " + scenes_string)
