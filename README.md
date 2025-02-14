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

Datasets:
[mipnerf360pt1](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip), 
[mipnerf360pt2](https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip), 
[zipnerf-undistorted](https://storage.googleapis.com/gresearch/refraw360/zipnerf-undistorted/checkpoints.zip), 
[zipnerf](https://storage.googleapis.com/gresearch/refraw360/zipnerf/checkpoints.zip)

`zipnerf-undistorted` is used for evaluation against 3DGS.

More details can be found in our [paper](https://arxiv.org/abs/2410.01804) or at our [website](https://half-potato.gitlab.io/posts/ever/)

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>
@misc{mai2024everexactvolumetricellipsoid, title={EVER: Exact Volumetric Ellipsoid Rendering for Real-time View Synthesis},  author={Alexander Mai and Peter Hedman and George Kopanas and Dor Verbin and David Futschik and Qiangeng Xu and Falko Kuester and Jon Barron and Yinda Zhang}, year={2024}, eprint={2410.01804}, archivePrefix={arXiv}, primaryClass={cs.CV}, url={https://arxiv.org/abs/2410.01804},  }
</code></pre>
  </div>
</section>


## Quick Install

### Dependencies
- OptiX 7.4, which must be downloaded from NVIDIA's [website](https://developer.nvidia.com/designworks/optix/downloads/legacy). This is downloaded and placed somewhere on your computer, then use `export OptiX_INSTALL_DIR=...` to set the variable to that location.
- [*SlangD*](https://github.com/shader-slang/slang). We recommend using the latest version you can, as they have fixed quite a few bugs. 
We can install the rest of the dependencies as follows:
```
sudo apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev
conda env create --name ever python==3.10
conda activate ever
conda install pip
# adjust for cuda version
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```
<!-- conda env create --file environment.yml -->

For Manjaro, Arch, or other rolling release, run the following:
```
export CXX=/usr/bin/g++-11 CC=/usr/bin/gcc-11 
```

Now, download the files and run `bash install.bash`
```
git clone --recursive https://github.com/half-potato/ever_training
cd ever_training
bash install.bash
```
If you get a bunch of compilation errors, it could be that you need to run the export line for the CXX and CC versions.

We can now train using the following command:
```
python train.py -s <path to COLMAP or NeRF Synthetic dataset>
```

Tested on Manjaro and Ubuntu Linux 22.04.

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
