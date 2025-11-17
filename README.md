# MCFNet
MCFNet: A Multi-scale Cross-modal Fusion Network for RGB-D Salient Object Detection

# Environments

```bash
conda create -n magnet python=3.9.18
conda activate magnet
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install -c conda-forge opencv-python==4.7.0
pip install timm==0.6.5
conda install -c conda-forge tqdm
conda install yacs
```

# Data Preparation

- Download the RGB-D raw data from [Google drive](https://drive.google.com/drive/folders/1-WqJBL74t1mvu62f29iwEHzsLTCbahQh?usp=sharing) <br>

Note that the depth maps of the raw data above are foreground is white.

# Training & Testing

- Train the MCFNet:
    1. download the pretrained PVT pth from [Google drive](https://drive.google.com/file/d/1HQVhygEP64vwFQRf4LnF7yyrZfBYfCHy/view?usp=sharing).
    2. modify the `rgb_root` `depth_root` `gt_root` in `train_Net.py` according to your own data path.
    3. run `python train_Net.py`
- Test the MCFNet:
    1. modify the `test_path` `pth_path` in `test_Net.py` according to your own data path.
    2. run `python test_Net.py`

# Evaluate tools

- You can select one of toolboxes to get the metrics
[CODToolbox](https://github.com/DengPingFan/CODToolbox) / [SOD_Evaluation_Metrics](https://github.com/zyjwuyan/SOD_Evaluation_Metrics)

# Saliency Maps

We provide the saliency maps of DUT, LFSD, NJU2K, NLPR, SIP, STERE datasets.

- RGB-D [Google drive].(https://drive.google.com/drive/folders/1xusyfPTMF2i-qTYHL7EpqT01R_xzfsSt?usp=sharing).<br>


# Trained Models

- RGB-D [Google drive].(https://drive.google.com/file/d/1RmPd27BkjRbTtZdFijSOIFa4KS8h4sWs/view?usp=sharing). <br>
