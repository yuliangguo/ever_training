#!/bin/bash
pip install -r requirements.txt

cp ever/new_files/*.py .
cp -r ever/new_files/notebooks .
cp ever/new_files/scene/* scene/
cp ever/new_files/gaussian_renderer/* gaussian_renderer/
cp ever/new_files/utils/* utils/

git apply ../ever/new_files/sibr_patch.patch

export OptiX_INSTALL_DIR=/home/yuliang/Tools/NVIDIA-OptiX-SDK-7.4.0-linux64-x86_64
export PATH=$PATH:/home/yuliang/Tools/slang-2025.5-linux-x86_64/bin
export CXX=/usr/bin/g++-11
export CC=/usr/bin/gcc-11

# Build splinetracer
mkdir ever/build
cd ever/build
# CXX=/usr/bin/g++-11 CC=/usr/bin/gcc-11 cmake -DOptiX_INSTALL_DIR=$OptiX_INSTALL_DIR -D_GLIBCXX_USE_CXX11_ABI=1 ..
CXX=$CXX CC=$CC cmake -DOptiX_INSTALL_DIR=$OptiX_INSTALL_DIR ..
make -j8
cd ../..

pip install -e submodules/simple-knn

# # SIBR Viewer
# cd SIBR_viewers
# cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release
# cmake --build build -j24 --target install
# cd ../..
