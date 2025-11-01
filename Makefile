.ONESHELL:
SHELL := /bin/bash
.DEFAULT_GOAL := help
# .PHONY: all help clean test_pangolin test_human pangolin_test
# This tells make to always run the recipe for the requested targets,
# regardless of whether a file with that name exists, thus no needs for .PHONY declaration.
MAKEFLAGS += --always-make

# Use ENV_VAR if set, otherwise default to "default_value"
CONDA_PREFIX?=$(HOME)/miniconda3
CONDA_SH_PATH?=$(CONDA_PREFIX)/etc/profile.d/conda.sh
CONDA_ENV_NAME?=ambisplice

debug=false
gpus=0
predict_size=20000

help: ## Display this help message
	@echo "Usage: make <target>"
	@echo
	@echo "Available targets:"
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-42s\033[0m %s\n", $$1, $$2}'

install: ## Install python dependencies under conda environment
	source $(CONDA_SH_PATH)
	conda create -n $(CONDA_ENV_NAME) python=3.11 -y
	conda activate $(CONDA_ENV_NAME)
	# Install PyTorch with CUDA 12.6 support
	pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
	pip install lightning
	conda install -c conda-forge pandas numpy hydra-core omegaconf wandb gputil matplotlib beartype h5py pytables -y

download_pangolin_genomes: ## Download Pangolin genomes
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

eval_pangolinomni2_pangolinsolo123: ## Evaluate PangolinOmni2 model trained on PangolinSolo123 dataset
	ckpt_dir=checkpoints/pangolinomni2.pangolinsolo123_2025-10-27_07-27-47_933otwx4
	tissues=(liver)
	tissues=(heart liver brain testis)
	tissues=(heart liver brain testis "heart,liver,brain,testis")	
	for ((i=0; i<$${#tissues[@]}; i++)) ; do
		conda run --no-capture-output --name $(CONDA_ENV_NAME) \
		python -u run_ambisplice.py stage=eval \
			infer.save_prefix=$${ckpt_dir}/pangolin_test_$${tissues[i]//,/-} \
			infer.save_level=2 infer.eval_dim=null \
			model.type=pangolinomni2 \
			model.state_dict_path=null \
			litrun.resume_from_ckpt=$${ckpt_dir}/last.ckpt \
			dataset.type=pangolinsolo \
			dataset.train_path=null \
			dataset.predict_size=$(predict_size) \
			dataset.predict_path=data/pangolin/dataset_test_1.h5 \
			+dataset.tissue_types=[$${tissues[i]}] \
			+dataset.tissue_embedding_path="data/tissue_avg_pca_embeddings.csv" \
			debug=$(debug)
		wait
	done

eval_pangolinomni_pangolinsolo123: ## Evaluate PangolinOmni model trained on PangolinSolo123 dataset
	ckpt_dir=checkpoints/pangolinomni.pangolinsolo123_2025-10-21_23-25-37_bsi1y0zt
	tissues=(liver)
	tissues=(heart liver brain testis "heart,liver,brain,testis")
	tissues=("heart,liver,brain,testis")	
	for ((i=0; i<$${#tissues[@]}; i++)) ; do
		conda run --no-capture-output --name $(CONDA_ENV_NAME) \
		python -u run_ambisplice.py stage=eval \
			infer.save_prefix=$${ckpt_dir}/pangolin_test_$${tissues[i]//,/-} \
			infer.save_level=2 infer.eval_dim=null \
			model.type=pangolinomni \
			model.state_dict_path=null \
			litrun.resume_from_ckpt=$${ckpt_dir}/last.ckpt \
			dataset.type=pangolinsolo \
			dataset.train_path=null \
			dataset.predict_size=$(predict_size) \
			dataset.predict_path=data/pangolin/dataset_test_1.h5 \
			+dataset.tissue_types=[$${tissues[i]}] \
			+dataset.tissue_embedding_path="data/tissue_avg_pca_embeddings.csv" \
			debug=$(debug)
		wait
	done

eval_pangolinsolo_pangolinsolo4: ## Evaluate PangolinSolo model trained on PangolinSolo testis dataset (single tissue input and output)
	ckpt_dir=checkpoints/pangolinsolo.pangolinsolo4_2025-10-27_16-15-32_8l8omvzn
	tissues=(liver)
	tissues=(heart liver brain testis "heart,liver,brain,testis")
	for ((i=0; i<$${#tissues[@]}; i++)) ; do
		conda run --no-capture-output --name $(CONDA_ENV_NAME) \
		python -u run_ambisplice.py stage=eval \
			infer.save_prefix=$${ckpt_dir}/pangolin_test_$${tissues[i]//,/-} \
			infer.save_level=2 infer.eval_dim=null \
			model.type=pangolinsolo \
			model.state_dict_path=null \
			litrun.resume_from_ckpt=$${ckpt_dir}/last.ckpt \
			dataset.type=pangolinsolo \
			dataset.train_path=null \
			dataset.predict_size=$(predict_size) \
			dataset.predict_path=data/pangolin/dataset_test_1.h5 \
			+dataset.tissue_types=[$${tissues[i]}] \
			debug=$(debug)
		wait
	done

eval_pangolinsolo_pangolinsolo3: ## Evaluate PangolinSolo model trained on PangolinSolo testis dataset (single tissue input and output)
	ckpt_dir=checkpoints/pangolinsolo.pangolinsolo3_2025-10-29_03-43-28_l9t6a1nc
	tissues=(liver)
	tissues=(heart liver brain testis "heart,liver,brain,testis")
	for ((i=0; i<$${#tissues[@]}; i++)) ; do
		conda run --no-capture-output --name $(CONDA_ENV_NAME) \
		python -u run_ambisplice.py stage=eval \
			infer.save_prefix=$${ckpt_dir}/pangolin_test_$${tissues[i]//,/-} \
			infer.save_level=2 infer.eval_dim=null \
			model.type=pangolinsolo \
			model.state_dict_path=null \
			litrun.resume_from_ckpt=$${ckpt_dir}/last.ckpt \
			dataset.type=pangolinsolo \
			dataset.train_path=null \
			dataset.predict_size=$(predict_size) \
			dataset.predict_path=data/pangolin/dataset_test_1.h5 \
			+dataset.tissue_types=[$${tissues[i]}] \
			debug=$(debug)
		wait
	done

eval_pangolinsolo_pangolinsolo2: ## Evaluate PangolinSolo model trained on PangolinSolo liver dataset (single tissue input and output)
	ckpt_dir=checkpoints/pangolinsolo.pangolinsolo2_2025-10-29_05-02-24_4wltwlym
	tissues=(liver)
	tissues=(heart liver brain testis "heart,liver,brain,testis")
	for ((i=0; i<$${#tissues[@]}; i++)) ; do
		conda run --no-capture-output --name $(CONDA_ENV_NAME) \
		python -u run_ambisplice.py stage=eval \
			infer.save_prefix=$${ckpt_dir}/pangolin_test_$${tissues[i]//,/-} \
			infer.save_level=2 infer.eval_dim=null \
			model.type=pangolinsolo \
			model.state_dict_path=null \
			litrun.resume_from_ckpt=$${ckpt_dir}/last.ckpt \
			dataset.type=pangolinsolo \
			dataset.train_path=null \
			dataset.predict_size=$(predict_size) \
			dataset.predict_path=data/pangolin/dataset_test_1.h5 \
			+dataset.tissue_types=[$${tissues[i]}] \
			debug=$(debug)
		wait
	done

eval_pangolinsolo_pangolinsolo1: ## Evaluate PangolinSolo model trained on PangolinSolo heart dataset (single tissue input and output)
# 	ckpt_dir=checkpoints/pangolinsolo_pangolin_2025-10-18_12-41-38_btn1jd24
	ckpt_dir=checkpoints/pangolinsolo.pangolinsolo1_2025-10-25_11-46-07_cqw6jfnv
	tissues=(liver)
	tissues=(heart liver brain testis)
	tissues+=("heart,liver,brain,testis")
	for ((i=0; i<$${#tissues[@]}; i++)) ; do
		conda run --no-capture-output --name $(CONDA_ENV_NAME) \
		python -u run_ambisplice.py stage=eval \
			infer.save_prefix=$${ckpt_dir}/pangolin_test_$${tissues[i]//,/-} \
			infer.save_level=2 infer.eval_dim=null \
			model.type=pangolinsolo \
			model.state_dict_path=null \
			litrun.resume_from_ckpt=$${ckpt_dir}/last.ckpt \
			dataset.type=pangolinsolo \
			dataset.train_path=null \
			dataset.predict_size=$(predict_size) \
			dataset.predict_path=data/pangolin/dataset_test_1.h5 \
			+dataset.tissue_types=[$${tissues[i]}] \
			debug=$(debug)
		wait
	done

eval_pangolinorig_ensemble_pangolin: ## Evaluate Pangolin original ensemble average
	# Run Pangolin model (final.modelnum.tissue.epoch) on Pangolin dataset
	model_dir=benchmarks/pangolin_models
	conda run --no-capture-output --name $(CONDA_ENV_NAME) \
	python -u run_ambisplice.py stage=eval \
	    infer.save_prefix=benchmarks/pangolin_ensemble/pangolin_test \
		infer.save_level=2 infer.eval_dim=-2 \
		model.type=pangolin \
		model.state_dict_path=null \
		litrun.resume_from_ckpt=null \
		ensemble.enable=true \
		ensemble.model.type=[pangolin,pangolin,pangolin] \
		ensemble.model.state_dict_path=[$${model_dir}/final.1.0.3,$${model_dir}/final.2.0.3,$${model_dir}/final.3.0.3,$${model_dir}/final.4.0.3,$${model_dir}/final.5.0.3] \
	    dataset.type=pangolin \
		dataset.predict_path=data/pangolin/dataset_test_1.h5 \
		dataset.predict_size=$(predict_size) \
		+dataset.tissue_types=[heart,liver,brain,testis] \
		debug=$(debug)

eval_pangolin_pangolin: ## Evaluate Pangolin model trained on Pangolin dataset (quad inputs and outputs)
	ckpt_dir="checkpoints/pangolin.pangolin_2025-10-21_22-16-17_mnsgbck4"
	conda run --no-capture-output --name $(CONDA_ENV_NAME) \
	python -u run_ambisplice.py stage=eval \
		infer.save_prefix=$${ckpt_dir}/pangolin_test \
		infer.save_level=2 infer.eval_dim=-2 \
		model.type=pangolin \
		model.state_dict_path=null \
		litrun.resume_from_ckpt=$${ckpt_dir}/last.ckpt \
		dataset.type=pangolin \
		dataset.train_path=null \
		dataset.predict_size=$(predict_size) \
		dataset.predict_path=data/pangolin/dataset_test_1.h5 \
		+dataset.tissue_types=[heart,liver,brain,testis] \
		debug=$(debug)

eval_pangolinorig_pangolin: ## Evaluate Pangolin original model
	# Run Pangolin model (final.modelnum.tissue.epoch) on Pangolin dataset
	#	dataset.predict_path=data/pangolin/dataset_train_all.h5 \
	models=(final.1.0.3 final.2.0.3 final.3.0.3 final.4.0.3 final.5.0.3 final.1.0.3.v2 final.2.0.3.v2 final.3.0.3.v2)
	models=(final.1.1.3 final.2.1.3 final.3.1.3 final.4.1.3 final.5.1.3 final.1.1.3.v2 final.2.1.3.v2 final.3.1.3.v2)
	models+=(final.1.2.3 final.2.2.3 final.3.2.3 final.4.2.3 final.5.2.3 final.1.2.3.v2 final.2.2.3.v2 final.3.2.3.v2)	
	models=(final.1.3.3 final.2.3.3 final.3.3.3 final.4.3.3 final.5.3.3 final.1.3.3.v2 final.2.3.3.v2 final.3.3.3.v2)
	models+=(final.1.4.3 final.2.4.3 final.3.4.3 final.4.4.3 final.5.4.3 final.1.4.3.v2 final.2.4.3.v2 final.3.4.3.v2)
	for model in $${models[@]} ; do
		conda run --no-capture-output --name $(CONDA_ENV_NAME) \
		python -u run_ambisplice.py stage=eval \
			infer.save_prefix=benchmarks/pangolin_$${model}/pangolin_test \
			infer.save_level=2 infer.eval_dim=-2 \
			model.type=pangolin \
			model.state_dict_path=benchmarks/pangolin_models/$${model} \
			litrun.resume_from_ckpt=null \
			ensemble.enable=false \
			dataset.type=pangolin \
			dataset.predict_path=data/pangolin/dataset_test_1.h5 \
			dataset.predict_size=$(predict_size) \
			+dataset.tissue_types=[heart,liver,brain,testis] \
			debug=$(debug)
	done

train_pangolinsolo_pangolinsolo123: ## Train PangolinSolo model on PangolinSolo dataset
	conda run --no-capture-output --name $(CONDA_ENV_NAME) \
	python -u run_ambisplice.py stage=train gpus=[$(gpus)] \
		run_name=pangolinsolo.pangolinsolo123 \
		model.type=pangolinsolo \
		model.state_dict_path=null \
		dataset.type=pangolinsolo \
		+dataset.tissue_types=[heart,liver,brain] \
		dataset.train_path=data/pangolin/dataset_train_all.h5 \
		dataloader.train_batch_size=96 \
		dataloader.val_batch_size=128 \
		litrun.resume_from_ckpt=null \
		debug=$(debug)

train_pangolinsolo_pangolinsolo124: ## Train PangolinSolo model on PangolinSolo dataset
	conda run --no-capture-output --name $(CONDA_ENV_NAME) \
	python -u run_ambisplice.py stage=train gpus=[$(gpus)] \
		run_name=pangolinsolo.pangolinsolo124 \
		model.type=pangolinsolo \
		model.state_dict_path=null \
		dataset.type=pangolinsolo \
		+dataset.tissue_types=[heart,liver,testis] \
		dataset.train_path=data/pangolin/dataset_train_all.h5 \
		dataloader.train_batch_size=96 \
		dataloader.val_batch_size=128 \
		litrun.resume_from_ckpt=null \
		debug=$(debug)

train_pangolinsolo_pangolinsolo1: ## Train PangolinSolo model on PangolinSolo dataset
	conda run --no-capture-output --name $(CONDA_ENV_NAME) \
	python -u run_ambisplice.py stage=train gpus=[$(gpus)] \
		run_name=pangolinsolo.pangolinsolo1 \
		model.type=pangolinsolo \
		model.state_dict_path=null \
		dataset.type=pangolinsolo \
		+dataset.tissue_types=[heart] \
		dataset.train_path=data/pangolin/dataset_train_all.h5 \
		dataloader.train_batch_size=96 \
		dataloader.val_batch_size=128 \
		litrun.resume_from_ckpt=null \
		debug=$(debug)

train_pangolinsolo_pangolinsolo2: ## Train PangolinSolo model on PangolinSolo 2 dataset
	conda run --no-capture-output --name $(CONDA_ENV_NAME) \
	python -u run_ambisplice.py stage=train gpus=[$(gpus)] \
		run_name=pangolinsolo.pangolinsolo2 \
		model.type=pangolinsolo \
		model.state_dict_path=null \
		dataset.type=pangolinsolo \
		+dataset.tissue_types=[liver] \
		dataset.train_path=data/pangolin/dataset_train_all.h5 \
		dataloader.train_batch_size=96 \
		dataloader.val_batch_size=128 \
		litrun.resume_from_ckpt=null \
		debug=$(debug)

train_pangolinsolo_pangolinsolo3: ## Train PangolinSolo model on PangolinSolo dataset
	conda run --no-capture-output --name $(CONDA_ENV_NAME) \
	python -u run_ambisplice.py stage=train gpus=[$(gpus)] \
		run_name=pangolinsolo.pangolinsolo3 \
		model.type=pangolinsolo \
		model.state_dict_path=null \
		dataset.type=pangolinsolo \
		+dataset.tissue_types=[brain] \
		dataset.train_path=data/pangolin/dataset_train_all.h5 \
		dataloader.train_batch_size=96 \
		dataloader.val_batch_size=128 \
		litrun.resume_from_ckpt=null \
		debug=$(debug)

train_pangolinsolo_pangolinsolo4: ## Train PangolinSolo model on PangolinSolo dataset
	conda run --no-capture-output --name $(CONDA_ENV_NAME) \
	python -u run_ambisplice.py stage=train gpus=[$(gpus)] \
		run_name=pangolinsolo.pangolinsolo4 \
		model.type=pangolinsolo \
		model.state_dict_path=null \
		dataset.type=pangolinsolo \
		+dataset.tissue_types=[testis] \
		dataset.train_path=data/pangolin/dataset_train_all.h5 \
		dataloader.train_batch_size=96 \
		dataloader.val_batch_size=128 \
		litrun.resume_from_ckpt=null \
		debug=$(debug)
		
train_pangolinomni_pangolinsolo123: ## Train PangolinOmni model on PangolinSolo dataset
	conda run --no-capture-output --name $(CONDA_ENV_NAME) \
	python -u run_ambisplice.py stage=train gpus=[$(gpus)] \
		run_name=pangolinomni.pangolinsolo123 \
		model.type=pangolinomni \
		model.state_dict_path=null \
		dataset.type=pangolinsolo \
		dataset.train_path=data/pangolin/dataset_train_all.h5 \
		+dataset.tissue_types=[heart,liver,brain] \
		+dataset.tissue_embedding_path="data/tissue_avg_pca_embeddings.csv" \
		dataloader.train_batch_size=96 \
		dataloader.val_batch_size=128 \
		litrun.resume_from_ckpt=null \
		debug=$(debug)

train_pangolinomni2_pangolinsolo123: ## Train PangolinOmni2 model on PangolinSolo dataset
	conda run --no-capture-output --name $(CONDA_ENV_NAME) \
	python -u run_ambisplice.py stage=train gpus=[$(gpus)] \
		run_name=pangolinomni2.pangolinsolo123 \
		model.type=pangolinomni2 \
		model.state_dict_path=null \
		dataset.type=pangolinsolo \
		dataset.train_path=data/pangolin/dataset_train_all.h5 \
		+dataset.tissue_types=[heart,liver,brain] \
		+dataset.tissue_embedding_path="data/tissue_avg_pca_embeddings.csv" \
		dataloader.train_batch_size=96 \
		dataloader.val_batch_size=128 \
		litrun.resume_from_ckpt=null \
		debug=$(debug)

train_pangolin_pangolin: ## Train Pangolin model on Pangolin dataset (all four tissues)
	conda run --no-capture-output --name $(CONDA_ENV_NAME) \
	python -u run_ambisplice.py stage=train gpus=[$(gpus)] \
		model.type=pangolin \
		model.state_dict_path=null \
		dataset.type=pangolin \
		dataset.train_path=data/pangolin/dataset_train_all.h5 \
		+dataset.tissue_types=[heart,liver,brain,testis] \
		dataloader.train_batch_size=96 \
		dataloader.val_batch_size=128 \
		litrun.resume_from_ckpt=null \
		debug=$(debug)

train_splicesolo_genecrops: ## Train SpliceSolo on genecrops dataset
	conda run --no-capture-output --name $(CONDA_ENV_NAME) \
	python -u run_ambisplice.py stage=train gpus=[$(gpus)] \
		model.type=SpliceSolo \
		model.state_dict_path=null \
		dataset.type=genecrops \
		dataset.train_path=data/gwsplice_genecrops.h5 \
		dataloader.train_batch_size=96 \
		dataloader.val_batch_size=128 \
		litrun.resume_from_ckpt=null \
		debug=$(debug)

train_splicesolo_genesites: ## Train SpliceSolo on gene sites dataset
	conda run --no-capture-output --name $(CONDA_ENV_NAME) \
	python -u run_ambisplice.py stage=train gpus=[$(gpus)] \
		model.type=splicesolo \
		model.state_dict_path=null \
		dataset.type=genesites \
		dataset.train_path=data/gwsplice_genesites.pkl \
		+dataset.stratified_sampling=chrom \
		+dataset.weighted_sampling=len \
		+dataset.dynamic_weights=false \
		dataloader.train_batch_size=96 \
		dataloader.val_batch_size=128 \
		litrun.resume_from_ckpt=null \
		debug=$(debug)