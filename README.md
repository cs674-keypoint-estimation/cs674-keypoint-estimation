## CS674 Final Project

### Directions:

- Folder structure works on Linux/Mac
- To install dependencies:
  - Run `sh setup.sh` to setup a Conda environment with all required dependencies installed.
**NOTE**: We assume you have CUDA running on your machine. Please change the CUDA version dependent libraries in **setup.sh** as needed. By default, it assumes you have CUDA 12.1.
  - Alternatively,
    - Follow instructions in readme.md for SC3K-dependency installation
    - Install additional libraries as needed.
- In case you do not plan to use flash-attention, simply skip installing it. Disable usage by setting mode.`flash_attention : False` in **config.yaml**.
- Inside config.yaml:
  - Set `alt_scripts: False` for "vanilla" SC3K and `alt_scripts: True` to run modified scripts.
  - Set `encoder: ptv3` to use the Point Transformer V3 encoder backbone, and `encoder: pointnet` to use the PointNet encoder backbone.
  - Set `decoder: ptv3` to use the PTv3 decoder instead of the SC3K residual blocks. 
  - Set `decoder: sc3k` to use the default SC3K residual blocks for decoding.
  - Select **train** or **test** with appropriate pcd name and corresponding keypointnet number.

- To train the model,
  - Run `python train.py`
- To test the model,
  - Copy the generated **Best_****.pth** to the folder mentioned in **best_model_path** in **config.yaml** (in case not present by default)
  - Run `python test.py`

- Generated best models using the four methods - vanilla SC3K, modified SC3K with PointNet, modified SC3K with PTv3, modified SC3K with PTv3 (encoder + decoder), can be found under `experiments/generated_models`.
- Generated visualizations from our experiments can be found under `experiments/visualizations`.
- In case you wish to run this on Google Colab (Open3D support is limited, so no visualizations, but good for training), use **Colab_SC3K_w_PTv3.ipynb**.

### Breakdown of Effort

**Code written by Luis Barriga:**
- network_alt up to line 64.
- utils_alt up to line 180.
- splitter.py (inside splits folder)

**Code written by Evan Young:**
- network_alt.py lines 65-93 (note that Luis reused this code from Evan's residual_block.py which contains additional norms)
- residual_block.py
- semirand_pose_generator_v2.py
- split_generator.py

**Code written by Aylin Elmali:**
- PTv3_model.py lines 916-927
- network.py lines 308-322
- Visualizations for PTv3-based SC3K network

**Code written by Matthew James:**
- setup.sh
- network.py lines 33-61, and lines 288-300 (modified to include PTv3 to network)
- Colab_SC3K_w_PTV3.ipynb