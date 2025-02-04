#!/bin/bash
cp ever/new_files/*.py .
cp -r ever/new_files/notebooks .
cp ever/new_files/scene/* scene/
cp ever/new_files/gaussian_renderer/* gaussian_renderer/
cp ever/new_files/utils/* utils/

git apply ../ever/new_files/sibr_patch.patch

# Build splinetracer
mkdir ever/build
cd ever/build
cmake -DOptiX_INSTALL_DIR=$OptiX_INSTALL_DIR -D_GLIBCXX_USE_CXX11_ABI=1 ..
make -j8
cd ../..

pip install -e submodules/simple-knn

# SIBR Viewer
cd SIBR_viewers
cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j24 --target install
cd ../..
