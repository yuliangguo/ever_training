#!/bin/bash
cp ever/new_files/*.py .
cp -r ever/new_files/notebooks .
cp ever/new_files/scene/* scene/
cp ever/new_files/gaussian_renderer/* gaussian_renderer/
cp ever/new_files/utils/* utils/

cd SIBR_viewers
git apply ../ever/new_files/sibr_patch.patch
