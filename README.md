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

- Download the RGB-D raw data from [baidu](https://pan.baidu.com/s/10Y90OXUFoW8yAeRmr5LFnA?pwd=exwj) / [Google drive](https://drive.google.com/file/d/19HXwGJCtz0QdEDsEbH7cJqTBfD-CEXxX/view?usp=sharing) <br>

Note that the depth maps of the raw data above are foreground is white.

# Training & Testing

- Train the MCFNet:
    1. download the pretrained PVT pth from [baidu](https://pan.baidu.com/s/11bNtCS7HyjnB7Lf3RIbpFg?pwd=bxiw) / [Google drive](https://drive.google.com/file/d/1eNhQwUHmfjR-vVGY38D_dFYUOqD_pw-H/view?usp=sharing), and put it under  `ckps/smt/`.
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

- RGB-D <br>


# Trained Models

- RGB-D [baidu](https://pan.baidu.com/s/1RPMA5Z3liMoUlG0AWuGeRA?pwd=5aqf) / [Google drive](https://drive.google.com/file/d/1vb2Vcbz9bCjvaSwoRZjIi39ae5Ei1GVs/view?usp=sharing) <br>
