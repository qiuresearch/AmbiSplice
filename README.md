# AmbiSplice

## Install
conda create -n ambisplice python=3.11 -y

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url 
https://download.pytorch.org/whl/cu126

pip install lightning

conda install -c conda-forge pandas hydra-core omegaconf wandb gputil matplotlib beartype colorlog -y
