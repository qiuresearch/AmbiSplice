.ONESHELL:
SHELL := /bin/bash
.DEFAULT_GOAL := help

CONDA_SH_PATH=$(CONDA_PREFIX)/etc/profile.d/conda.sh
CONDA_ENV_NAME=ambisplice
# Use ENV_VAR if set, otherwise default to "default_value"
# ENV_VAR ?= default_value

.PHONY: all help clean test_pangolin test_human pangolin_test

help: ## Display this help message
	@echo "Usage: make <target>"
	@echo
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	source $(CONDA_SH_PATH) || exit 1
	conda create -n $(CONDA_ENV_NAME) python=3.11 -y || exit 1
	conda activate $(CONDA_ENV_NAME) || exit 1
	# Install PyTorch with CUDA 12.6 support
	pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
	pip install lightning
	conda install -c conda-forge pandas numpy hydra-core omegaconf wandb gputil matplotlib beartype h5py -y

pangolin_datasets: ## Prepare Pangolin datasets
	echo "Preparing Pangolin datasets..."
	# 

pangolin_download_genomes: ## Download Pangolin genomes

	wget https://ftp.ensembl.org/pub/release-108/fasta/macaca_mulatta/dna/Macaca_mulatta.Mmul_10.dna.toplevel.fa.gz
	wget https://ftp.ensembl.org/pub/release-99/fasta/rattus_norvegicus/dna/Rattus_norvegicus.Rnor_6.0.dna.toplevel.fa.gz


pangolin_test: ## Test Pangolin model
	# Run Pangolin model (final.modelnum.tissue.epoch) on Pangolin dataset
	conda run --no-capture-output --name $(CONDA_ENV_NAME) python -u run_ambisplice.py stage=eval \
	    save_prefix=Pangolin_model_test \
	    dataset.type=Pangolin \
		dataset.file_path=$(HOME)/github/Pangolin_train/preprocessing/dataset_train_all.h5 \
		model.type=Pangolin \
		model.state_dict_path=$(HOME)/github/Pangolin/pangolin/models/final.1.0.3.v2 \
		litrun.resume_from_ckpt=null \
		ensemble.enable=false

pangolin_train_single: ## Train Pangolin model
	# Run Pangolin model (final.modelnum.tissue.epoch) on Pangolin dataset
	conda run --no-capture-output --name $(CONDA_ENV_NAME) python -u run_ambisplice.py stage=train \
	    dataset.type=Pangolin \
		dataset.file_path=$(HOME)/github/Pangolin_train/preprocessing/dataset_train_all.h5 \
		model.type=Pangolin \
		model.state_dict_path=$(HOME)/github/Pangolin/pangolin/models/final.1.0.3.v2 \
		litrun.resume_from_ckpt=null

test_pangolin_dataset: ## Test on Pangolin dataset
	conda run --no-capture-output --name $(CONDA_ENV_NAME) python -u run_ambisplice.py stage=test \
		dataset.type=pangolin \
		dataset.file_path=$(HOME)/github/Pangolin_train/preprocessing/dataset_train_all.h5

test_human: ## Test on Human dataset
	conda run --no-capture-output --name $(CONDA_ENV_NAME) python -u run_ambisplice.py stage=test \
		dataset.type=ambisplice \
		dataset.file_path=$(HOME)/bench/AmbiSplice/data/ambisplice_test.h5

train_human_single: ## Train on Human Heart dataset
	conda run --no-capture-output --name $(CONDA_ENV_NAME) python -u run_ambisplice.py stage=train \
		dataset.type=ambisplice \
		dataset.file_path=rna_sites.pkl \
		litrun.max_epochs=10 \
		model.type=AmbiSplice \
		litrun.resume_from_ckpt=null \ # checkpoints/PangolinSingle_2025-10-07_06-49-35_eny45xtr/epoch=095-step=096000.ckpt