# Exact Volumetric Ellipsoid Rendering for Real-time View Synthesis
This is the repository with changes to 3DGS's training code to use the EVER rendering method.

Ever is a method for real-time differentiable emission-only volume rendering. Unlike recent
rasterization based approach by 3D Gaussian Splatting (3DGS), our primitive based representation
allows for exact volume rendering, rather than alpha compositing 3D Gaussian billboards. As such,
unlike 3DGS our formulation does not suffer from popping artifacts and view dependent density, but
still achieves frame rates of ∼30 FPS at 720p on an NVIDIA RTX4090. Because our approach is built
upon ray tracing it supports rendering techniques such as defocus blur and camera distortion (e.g.
such as from fisheye cameras), which are difficult to achieve by rasterization. We show that our
method has higher performance and fewer blending issues than 3DGS and other subsequent works,
especially on the challenging large-scale scenes from the Zip-NeRF dataset where it achieves SOTA
results among real-time techniques.

More details can be found in our (paper)[https://arxiv.org/abs/2410.01804] or at our (website)[https://half-potato.gitlab.io/posts/ever/]

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>
@misc{mai2024everexactvolumetricellipsoid, title={EVER: Exact Volumetric Ellipsoid Rendering for Real-time View Synthesis},  author={Alexander Mai and Peter Hedman and George Kopanas and Dor Verbin and David Futschik and Qiangeng Xu and Falko Kuester and Jon Barron and Yinda Zhang}, year={2024}, eprint={2410.01804}, archivePrefix={arXiv}, primaryClass={cs.CV}, url={https://arxiv.org/abs/2410.01804},  }
</code></pre>
  </div>
</section>


## Quick Install
First, download NVIDIA OptiX and put it somewhere like your home directory and use `export OptiX_INSTALL_DIR=...` to set the variable to that location.

Then:
```
sudo apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev
git submodule init
git submodule update
bash add_files.bash
conda env create --file environment.yml
conda activate gaussian_splatting
cd ever/build
cmake -DOptiX_INSTALL_DIR=$OptiX_INSTALL_DIR ..
make -j8
cd ../..
cd SIBR_viewers
cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j24 --target install
cd ../..
```
Tested on Manjaro and Ubuntu Linux 22.04.

## Optimizer

The optimizer uses PyTorch and CUDA extensions in a Python environment to produce trained models. 

### Hardware Requirements

- CUDA-ready GPU with Compute Capability 7.0+
- 24 GB VRAM (to train to paper evaluation quality)
- Please see FAQ for smaller VRAM configurations

### Software Requirements
- Conda or Mamba (Mamba is faster Conda)
- C++ Compiler for PyTorch extensions (we used Visual Studio 2019 for Windows)
- CUDA SDK 11 or 12 for PyTorch extensions, install *after* Visual Studio (we used 11.8, **known issues with 11.6**)
- C++ Compiler and CUDA SDK must be compatible (if not, use the following command: `export CC=/usr/bin/gcc-11 && export CXX=/usr/bin/g++-11` on Linux)
- OptiX 7.*, which must be downloaded from NVIDIA's website. This is downloaded and placed somewhere on your computer, which we will call $OptiX_INSTALL_DIR
- [*SlangD*](https://github.com/shader-slang/slang). We recommend using the latest version you can, as they have fixed quite a few bugs. 

### Evaluation
By default, the trained models use all available images in the dataset. To train them while withholding a test set for evaluation, use the ```--eval``` flag. This way, you can render training/test sets and produce error metrics as follows:
```shell
python train.py -s <path to COLMAP or NeRF Synthetic dataset> --eval # Train with train/test split --images (images_4 for 360 outdoor, images_2 for 360 indoor)
python render.py -m <path to trained model> # Generate renderings
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```

If you want to evaluate our [pre-trained models](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip), you will have to download the corresponding source data sets and indicate their location to ```render.py``` with an additional ```--source_path/-s``` flag. Note: The pre-trained models were created with the release codebase. This code base has been cleaned up and includes bugfixes, hence the metrics you get from evaluating them will differ from those in the paper.
```shell
python render.py -m <path to pre-trained model> -s <path to COLMAP dataset>
python metrics.py -m <path to pre-trained model>
```

These have the same arguments as 3DGS.

## Interactive Viewers
For all viewing purposes, we rely on the [SIBR](https://sibr.gitlabpages.inria.fr/) remote viewer. The training script will host a server to view training, and we provide the `host_render_server.py` file for viewing trained models.

### Hardware Requirements
- OpenGL 4.5-ready GPU and drivers (or latest MESA software)
- 4 GB VRAM recommended
- CUDA-ready GPU with Compute Capability 7.0+ (only for Real-Time Viewer)

### Software Requirements
- Visual Studio or g++, **not Clang** (we used Visual Studio 2019 for Windows)
- CUDA SDK 11, install *after* Visual Studio (we used 11.8)
- CMake (recent version, we used 3.24)
- OptiX 7.*, which must be downloaded from NVIDIA's website. This is downloaded and placed somewhere on your computer, which we will call $OptiX_INSTALL_DIR
- 7zip (only on Windows)

The viewer can then be run as follows:
```
python host_render_server.py -m $TRAINED_MODEL_LOCATION -s $SCENE_LOCATION --port $PORT --ip $IP`
```
The `$SCENE_LOCATION` only needs to be provided if viewing on a different machine than the model was trained on. `$IP` is for viewing on remote machines. By default, it is `127.0.0.1`. By default, `$PORT` is 6009. Once the render server has been hosted, we can then run the SIBR remote viewer in a separate terminal and connect it.
Then, on a different terminal, run:
```
./install/bin/SIBR_remoteGaussian_app --ip $IP --port $PORT
```


<details>
<summary><span style="font-weight: bold;">Primary Command Line Arguments for Network Viewer</span></summary>

  #### --path / -s
  Argument to override model's path to source dataset.
  #### --ip
  IP to use for connection to a running training script.
  #### --port
  Port to use for connection to a running training script. 
  #### --rendering-size 
  Takes two space separated numbers to define the resolution at which network rendering occurs, ```1200``` width by default.
  Note that to enforce an aspect that differs from the input images, you need ```--force-aspect-ratio``` too.
  #### --load_images
  Flag to load source dataset images to be displayed in the top view for each camera.
</details>
<br>
