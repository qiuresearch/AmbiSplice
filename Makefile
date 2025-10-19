.ONESHELL:
SHELL := /bin/bash
.DEFAULT_GOAL := help
debug=false

# Use ENV_VAR if set, otherwise default to "default_value"
# ENV_VAR ?= default_value
CONDA_PREFIX?=$(HOME)/miniconda3
CONDA_SH_PATH=$(CONDA_PREFIX)/etc/profile.d/conda.sh
CONDA_ENV_NAME=ambisplice

.PHONY: all help clean test_pangolin test_human pangolin_test

help: ## Display this help message
	@echo "Usage: make <target>"
	@echo
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	source $(CONDA_SH_PATH)
	conda create -n $(CONDA_ENV_NAME) python=3.11 -y
	conda activate $(CONDA_ENV_NAME)
	# Install PyTorch with CUDA 12.6 support
	pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
	pip install lightning
	conda install -c conda-forge pandas numpy hydra-core omegaconf wandb gputil matplotlib beartype h5py pytables -y

pangolin_download_genomes: ## Download Pangolin genomes
	# Download human and mouse genomes from Gencode (or Ensembl)
	wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/GRCh38.primary_assembly.genome.fa.gz
	gunzip GRCh38.primary_assembly.genome.fa.gz &
	wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/GRCm38.primary_assembly.genome.fa.gz
	gunzip GRCm38.primary_assembly.genome.fa.gz &
# 	wget https://ftp.ensembl.org/pub/release-115/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
# 	wget https://ftp.ensembl.org/pub/release-99/fasta/mus_musculus/dna/Mus_musculus.GRCm38.dna.primary_assembly.fa.gz

	wget https://ftp.ensembl.org/pub/release-108/fasta/macaca_mulatta/dna/Macaca_mulatta.Mmul_10.dna.toplevel.fa.gz
	gunzip Macaca_mulatta.Mmul_10.dna.toplevel.fa.gz &
	wget https://ftp.ensembl.org/pub/release-99/fasta/rattus_norvegicus/dna/Rattus_norvegicus.Rnor_6.0.dna.toplevel.fa.gz
	gunzip Rattus_norvegicus.Rnor_6.0.dna.toplevel.fa.gz &
	
	wait

	faidx GRCh38.primary_assembly.genome.fa
	faidx GRCm38.primary_assembly.genome.fa
	faidx Macaca_mulatta.Mmul_10.dna.toplevel.fa
	faidx Rattus_norvegicus.Rnor_6.0.dna.toplevel.fa

pangolin_eval: ## Evaluate Pangolin model
	# Run Pangolin model (final.modelnum.tissue.epoch) on Pangolin dataset
	conda run --no-capture-output --name $(CONDA_ENV_NAME) python -u run_ambisplice.py stage=eval \
	    save_prefix=Pangolin_model_test \
	    dataset.type=Pangolin \
		dataset.predict_path=$(HOME)/github/Pangolin_train/preprocessing/dataset_train_all.h5 \
		model.type=Pangolin \
		model.state_dict_path=$(HOME)/github/Pangolin/pangolin/models/final.1.0.3.v2 \
		litrun.resume_from_ckpt=null \
		ensemble.enable=false debug=$(debug)

pangolin_ensemble_eval: ## Evaluate Pangolin ensemble average
	# Run Pangolin model (final.modelnum.tissue.epoch) on Pangolin dataset
	conda run --no-capture-output --name $(CONDA_ENV_NAME) python -u run_ambisplice.py stage=eval \
	    save_prefix=Pangolin_ens_test \
	    dataset.type=Pangolin \
		dataset.predict_path=$(HOME)/github/Pangolin_train/preprocessing/dataset_train_all.h5 \
		model.type=Pangolin \
		model.state_dict_path=null \
		ensemble.enable=true \
		ensemble.model.type=[Pangolin,Pangolin,Pangolin] \
		ensemble.model.state_dict_path=[$(HOME)/github/Pangolin/pangolin/models/final.1.0.3.v2,$(HOME)/github/Pangolin/pangolin/models/final.2.0.3.v2,$(HOME)/github/Pangolin/pangolin/models/final.3.0.3.v2] \
		litrun.resume_from_ckpt=null debug=$(debug)

pangolin_train_single: ## Train Pangolin model in Single mode
	conda run --no-capture-output --name $(CONDA_ENV_NAME) python -u run_ambisplice.py stage=train \
	    dataset.type=Pangolin \
		dataset.train_path=data/pangolin/dataset_train_all.h5 \
		model.type=PangolinSingle \
		model.state_dict_path=null \
		dataloader.train_batch_size=96 \
		dataloader.val_batch_size=128 \
		litrun.resume_from_ckpt=null debug=$(debug)

test_pangolin_dataset: ## Test on Pangolin dataset
	conda run --no-capture-output --name $(CONDA_ENV_NAME) python -u run_ambisplice.py stage=test \
		dataset.type=pangolin \
		dataset.predict_path=$(HOME)/github/Pangolin_train/preprocessing/dataset_train_all.h5 \
		debug=$(debug)

test_human: ## Test on Human dataset
	conda run --no-capture-output --name $(CONDA_ENV_NAME) python -u run_ambisplice.py stage=test \
		dataset.type=ambisplice \
		dataset.predict_path=$(HOME)/bench/AmbiSplice/data/ambisplice_test.h5 \
		debug=$(debug)

train_genecrops_splicesingle: ## Train SpliceSingle on gene crops
	conda run --no-capture-output --name $(CONDA_ENV_NAME) python -u run_ambisplice.py stage=train \
		dataset.type=GeneCrops \
		dataset.train_path=data/gwsplice_genecrops.h5 \
		model.type=SpliceSingle \
		dataloader.train_batch_size=96 \
		dataloader.val_batch_size=128 \
		litrun.resume_from_ckpt=null debug=$(debug)