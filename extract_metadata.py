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
from pathlib import Path
from argparse import ArgumentParser, Namespace
import pickle
from PIL import Image, ExifTags, TiffImagePlugin
import subprocess
import json
from tqdm import tqdm

parser = ArgumentParser(description="Extract metadata parameters")
parser.add_argument('images', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--exif_tool_path', type=str, default="exiftool")
args = parser.parse_args()

metadatas = {
}

iso_tags = ["Sony ISO", "ISO"]
exposure_tags = ["Sony Exposure Time 2", "Exposure Time"]
aperature_tags = ["FNumber", "Sony F Number 2"]

def get_value(data, tags):
    vs = [data[t] for t in tags if t in data]
    return vs[0] if len(vs) > 0 else -1

for path in tqdm(args.images.iterdir()):
    try:
        img = Image.open(path)
    except:
        print(path, " is not an image")

    process = subprocess.Popen(
        [args.exif_tool_path,str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True) 
    exif = {
        ExifTags.TAGS[k]: float(v) if isinstance(v, TiffImagePlugin.IFDRational) else v
        for k, v in img._getexif().items()
        if k in ExifTags.TAGS
    }
    for tag in process.stdout:
        line = tag.strip().split(':')
        exif[line[0].strip()] = line[-1].strip()
    data = dict(
        iso=get_value(exif, iso_tags),
        exposure=get_value(exif, exposure_tags),
        aperature=get_value(exif, aperature_tags),
    )
    # print(exif)
    metadatas[path.name] = data

with args.output.open("w") as f:
    json.dump(metadatas, f)

