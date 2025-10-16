# AmbiSplice

## Install
conda create -n ambisplice python=3.11 -y

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url 
https://download.pytorch.org/whl/cu126

pip install lightning

conda install -c conda-forge pandas numpy condahydra-core omegaconf wandb gputil matplotlib beartype h5py pytables -y

## Datasets
### Pangolin original data
1. Pangolin_train github repo provides the following intermediate data files:
- splice_table_Human.test.txt  
- splice_table_Human.txt  
- splice_table_Macaque.txt  
- splice_table_Mouse.txt  
- splice_table_Rat.txt

We can use them to generate training data instead of repeating all data processing pipelines starting from RNA-seq raw reads.

2. Ideally, one only needs to run one script: create_files.sh in the preprocessing folder. Several genome sequence files and software tools may need to be downloaded or installed.

### Spliser processing

### Majiq processing

