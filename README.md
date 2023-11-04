# 3D Scene Synthesis

## Install
1. Use [conda]() to deploy the environment by
    ```commandline
    cd ScenePriors
    conda create env -f environment.yml
    conda activate sceneprior
    ```

2. Install [Fast Transformers](https://fast-transformers.github.io/) by
   ```commandline
   cd external/fast_transformers
   python setup.py build_ext --inplace
   cd ../..
   ```

3. Please follow [link](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) to install the prerequisite libraries for [PyTorch3D](https://pytorch3d.org/). Then install PyTorch3D from the local clone by
   ```commandline
   cd external/pytorch3d
   pip install -e .
   cd ../..
   ```
   *Note: After installed all prerequisite libraries in [link](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md), please do not install prebuilt binaries for PyTorch3D.*  
---

## Data Processing
### 3D-Front data processing (for scene genration)
1. Apply \& Download the [3D-Front](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) dataset and link them to the local directory as follows:
   ```
   datasets/3D-Front/3D-FRONT
   datasets/3D-Front/3D-FRONT-texture
   datasets/3D-Front/3D-FUTURE-model
   ```
   
2. Render 3D-Front scenes following [rendering pipeline](https://github.com/yinyunie/BlenderProc-3DFront) and link the rendering results (in `renderings` folder) to
   ```
   datasets/3D-Front/3D-FRONT_renderings_improved_mat
   ```
   *Note: you can comment out `bproc.renderer.enable_depth_output(activate_antialiasing=False)` in `render_dataset_improved_mat.py` since we do not need depth information.*

3. Preprocess 3D-Front data by
   ```commandline
   python utils/threed_front/1_process_viewdata.py --room_type ROOM_TYPE --n_processes NUM_THREADS
   python utils/threed_front/2_get_stats.py --room_type ROOM_TYPE
   ```
   * The processed data for training are saved in `datasets/3D-Front/3D-FRONT_samples`.
   * The parsed and extracted 3D-Front data for visualization is saved into `datasets/3D-Front/3D-FRONT_scenes`.
   * `ROOM_TYPE` can be `'bed'`(bedroom) or `'living'`(living room).
   * You can set `NUM_THREADS` to your CPU core number for parallel processing.

4. Visualize processed data for verification by (optional)
   ```commandline
   python utils/threed_front/vis/vis_gt_sample.py --scene_json SCENE_JSON_ID --room_id ROOM_ID --n_samples N_VIEWS 
   ```
   * `SCENE_JSON_ID` is the ID of a scene, e,g, `6a0e73bc-d0c4-4a38-bfb6-e083ce05ebe9`.
   * `ROOM_ID` is the room ID in this scene, e.g., `MasterBedroom-2679`.
   * `N_VIEWS` is the number views to visualize., e.g. `12`.

## Training
1. This project relies on [wandb](https://wandb.ai/site) to log the training process, you can log in with your API through [Get Started](https://wandb.ai/quickstart?utm_source=app-resource-center&utm_medium=app&utm_term=quickstart)
1. start layout pre-training by
    ```commandline
    python main.py \
       start_deform=False \
       resume=False \
       finetune=False \
       weight=[] \
       distributed.available_gpus=[6]\
       distributed.num_gpus=1 \
       data.dataset=3D-Front \
       data.split_type=bed \
       data.n_views=20 \
       data.aug=False \
       device.num_workers=32 \
       train.batch_size=128 \
       train.epochs=800 \
       scheduler.latent_input.milestones=[400] \
       scheduler.generator.milestones=[400] \
       log.if_wandb=True \
       exp_name=pretrain_3dfront_bedroom
    ```
   The network weight will be saved in `outputs/3D-Front/train/YEAR-MONTH-DAY/HOUR-MINUTE-SECOND/model_best.pth`. 
 
    The VAE-based model I defined is saved in `models/ours/modules/VAE.py`, and `models/ours/modules/extraction.py` shows the structure of convolutional layers in the VAE.

    With this command, the code runs on GPU 6. It can be changed by changing distribtued.available_gpus. It is recommended to use one GPU.

2. Shape training - We start shape training after the layout training converged. Please replace the `weight` keyword below with the pretrained weight path.
    ```commandline
    python main.py \
        start_deform=True \
        resume=False \
        finetune=True \
        weight=['outputs/3D-Front/train/YEAR-MONTH-DAY/HOUR-MINITE-SECOND/model_best.pth'] \
        distributed.available_gpus=[6]\
        distributed.num_gpus=1 \
        data.dataset=3D-Front \
        data.n_views=20 \
        data.aug=False \
        data.downsample_ratio=4 \
        device.num_workers=16 \
        train.batch_size=16 \
        train.epochs=500 \
        train.freeze=[] \
        scheduler.latent_input.milestones=[300] \
        scheduler.generator.milestones=[300] \
        log.if_wandb=True \
        exp_name=train_3dfront_bedroom 
    ```
    Still, the refined network weight will be saved in `outputs/3D-Front/train/YEAR-MONTH-DAY/HOUR-MINUTE-SECOND/model_best.pth`.

## Generation & Reconstruction
Please replace the keyword `weight` below with your trained weight path.
1. Scene Generation (with 3D-Front)
   ```commandline
   python main.py \
      mode=generation \
      start_deform=True \
      data.dataset=3D-Front \
      finetune=True \
      distributed.available_gpus=[6]\
      distributed.num_gpus=1 \
      weight=outputs/3D-Front/train/YEAR-MONTH-DAY/HOUR-MINUTE-SECOND/model_best.pth \
      generation.room_type=bed \
      data.split_dir=splits \
      data.split_type=bed \
      generation.phase=generation
   ```
   The generated scenes will be saved in `outputs/3D-Front/generation/YEAR-MONTH-DAY/HOUR-MINUTE-SECOND`.

## Visualization
Since there is no display for the server, X Server can be built by following the steps in prerequisites.
### Prerequisites

1. Install [Xming](https://sourceforge.net/projects/xming/) to mirror the virtual display.

### Runtime
1. Run the Xming.

2. Build a virtual display port on the server by
    ```
    Xephyr -br -ac -noreset -screen 800x600 :[port_number]
    ```
    where the port number is normally set to a number in [0, 100].

3. Set the environment variable `DISPLAY` to the corresponding display port number in last step. This can be done by adding
    ```python
    os.environ['DISPLAY']=':port_number'
    ```
    to the visualization files, as in the following files:
    |File|Description|
    |---|---|
    |./utils/threed_front/vis/vis_pred.py|Visualizing generated scene|
    |./utils/threed_front/vis/save_vis_gt.py|Saving images of ground truth scenes|
    |./utils/threed_front/vis/save_vis_pred.py|Saving images of generated scenes|

4. Run the visulization script by for example
   ```commandline
   python utils/threed_front/vis/vis_pred.py --pred_file outputs/3D-Front/generation/YEAR-MONTH-DAY/HOUR-MINUTE-SECOND/vis/bed/sample_X_X.npz --device 6 [--use_retrieval]
   ```
   where the scene saved in `sample_X_X.npz` is visualized with program running on device 6. By set the argument `use_retrieval`, the scene with retrieved meshes can be visualized.
## Evaluation
1. Save top-down orthographic projection of the scenes and objects class generated by ScenePriors and this work. The supervisior includes both 2D and 2D+3D.
    ```commandline
    python utils/threed_front/vis/save_vis_pred.py
    ```
    The generated images will be saved in `eva_image/pred_MODE` and the object class will be saved in `eva_image/pred_MODE.txt`.
2. Save the top-down orthographic projection and objects class of ground truth scenes.
    ```commandline
    python utils/threed_front/vis/save_vis_gt.py
    ```
    The generated images will be saved in `eva_image/gt` and the object class will be saved in `eva_image/gt_cls.txt`.
3. FID score
    Evaluate the FID score with the following:
    ```commandline
    python eva_FID.py
    ```
4. Category KL
    Evaluate the catergory KL with the following:
    ```commandline
    python eva_KL.py
    ```

## Pretrained Models

|    Model name      | Room type | Training stage |                      Path                     |                       Description                     |
| ------------------ | --------- | ------------- | --------------------------------------------- | ----------------------------------------------------- |
|ScenePriors_Layout  |  Bedroom  |Layout training|trained_model/ScenePriors_Layout/model_best.pth|ScenePriors layout training result                     |
|VAE_Layout          |  Bedroom  |Layout training|trained_model/VAE_Layout/model_best.pth        |VAE layout training result                             |
|ScenePriors_Shape   |  Bedroom  |Shape training |trained_model/ScenePriors_Shape                |ScenePriors shape training result                      |
|ScenePriors_Shape_CD|  Bedroom  |Shape training |trained_model/ScenePriors_Shape_CD             |ScenePriors shape training with Chamfer distance result|
|VAE_Shape           |  Bedroom  |Shape training |trained_model/VAE_Shape                        |VAE shape training result                              |
|VAE_Shape_CD        |  Bedroom  |Shape training |trained_model/VAE_Shape_CD                     |VAE shape training with Chamfer distance result        |

## Generated Scenes

|    Model name      | Room type |                Path              |             Description          |
| ------------------ | --------- | -------------------------------- | -------------------------------- |
|ScenePriors         |  Bedroom  |generated_scenes/ScenePriors      |ScenePriors                       |
|ScenePriors_CD      |  Bedroom  |generated_scenes/ScenePriors_CD   |ScenePriors with Chamfer distance |
|VAE                 |  Bedroom  |generated_scenes/VAE              |VAE                               |
|VAE_CD              |  Bedroom  |generated_scenes/VAE_CD           |VAE with Chamfer distance         |

