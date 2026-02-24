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

gpus=0
debug=false
sbatch=false
partition=small-gpu
time=7-00:00:00

help: ## Display this help message
	@echo "Usage: make <target>"
	@echo
	@echo "Available targets:"
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-42s\033[0m %s\n", $$1, $$2}'

sbatch_redirect: ## Redirect the action to sbatch instead of interactive running
	@if [ -n "$${SLURM_JOB_ID}" ] ; then exit 0 ; fi
	@GOALOPT="gpus=$(gpus) debug=$(debug)"
	@if [ "$(sbatch)" = "true" ] ; then \
		sbatch_file=sbatch ; sbatch_cmds=(echo Starting...) ; \
		for goal in $(MAKECMDGOALS) ; do sbatch_file=$${sbatch_file}_$${goal} ; sbatch_cmds+=(\; make $${goal} $${GOALOPT}) ; done ; \
		sbatch_brew.sh -p $(partition) -t $(time) -o "$${sbatch_file}.sh" "$${sbatch_cmds[*]}" ; \
		echo "Submitting via sbatch ..." ; \
		sbatch "$${sbatch_file}.sh" ; \
		exit 1 ; \
	fi

install: ## Install python dependencies under conda environment
	source $(CONDA_SH_PATH)
	conda create -n $(CONDA_ENV_NAME) python=3.11 -y
	conda activate $(CONDA_ENV_NAME)
	# Install PyTorch with CUDA 12.6 support
	pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
	pip install lightning
	pip install lightning[extra]
	conda install -c conda-forge pandas numpy hydra-core omegaconf wandb gputil matplotlib beartype h5py pytables -y
	conda install mappy -c bioconda -y

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

download_gencode_references: ## Download GENCODE reference genome and transcriptome
	wget -c --no-proxy -L https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/GRCh38.primary_assembly.genome.fa.gz
	wget -c --no-proxy -L https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.primary_assembly.annotation.gtf.gz
# 	gunzip GRCh38.primary_assembly.genome.fa.gz &
# 	wget --no-proxy -L https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/GRCm38.primary_assembly.genome.fa.gz
# 	gunzip GRCm38.primary_assembly.genome.fa.gz &
# 	wget https://ftp.ensembl.org/pub/release-115/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
# 	wget https://ftp.ensembl.org/pub/release-99/fasta/mus_musculus/dna/Mus_musculus.GRCm38.dna.primary_assembly.fa.gz

download_ensembl_references: ## Download ensembl reference genome and transcriptome
# 	latest_release=$$(curl --noproxy -s 'http://rest.ensembl.org/info/software?content-type=application/json' | grep -o '"release":[0-9]*' | cut -d: -f2)
	latest_release=115
	wget -c --no-proxy -L ftp://ftp.ensembl.org/pub/release-$${latest_release}/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz
	wget -c --no-proxy -L ftp://ftp.ensembl.org/pub/release-$${latest_release}/gtf/homo_sapiens/Homo_sapiens.GRCh38.$${latest_release}.gtf.gz

rnaseq_test_nextflow: ## run nextflow test on rnaseq data
	nextflow run nf-core/rnaseq -profile test_full,docker --outdir test

entex_rnaseq_fastq_nextflow: ## EnTEX dataset RNA-seq processing with nextflow
	data_dir=entex/downloads_sample_name
	nf_files=($$(ls $${data_dir}/P*_RNA-seq_fastq_nextflow.csv))

# 			--additional_fasta to add spike-in sequences (e.g. ERCC), but conflict with --transcript_fast option
# 			--fasta  Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz \
# 			--gtf Homo_sapiens.GRCh38.115.gtf.gz \
# run it once with save_reference and then use the saved gene_bed, transcriptome fasta, star and salmon indices
	for ((i=0; i<$${#nf_files[@]}; i++)) ; do
		tissue_type=$$(basename -s _RNA-seq_fastq_nextflow.csv $${nf_files[i]})
		echo "nextflow file: $${nf_files[i]}; tissue: $${tissue_type}"
		nextflow run nf-core/rnaseq -r 3.22.2 \
			-profile docker \
			--input $${nf_files[i]} \
			--outdir entex/RNA-seq_dataset/$${tissue_type} \
			--gencode \
			--fasta GRCh38.primary_assembly.genome.fa.gz \
			--gtf gencode.v49.primary_assembly.annotation.gtf.gz \
			--aligner star_salmon \
			--stringtie_ignore_gtf true \
			--save_reference \
			--save_align_intermeds
# 			--gene_bed \
# 			--transcript_fasta \
#           --rsem_index \
# 			--star_index \
# 			--salmon_index \
#           --max_cpus \
#           --skip_alignment	
#           -resume
# --extra_star_align_args "--alignIntronMax 1000000 --alignIntronMin 20 --alignMatesGapMax 1000000 --alignSJoverhangMin 8 --outFilterMismatchNmax 999 --outFilterMultimapNmax 20 --outFilterType BySJout --outFilterMismatchNoverLmax 0.1 --clip3pAdapterSeq AAAAAAAA"
# --extra_salmon_quant_args "--noLengthCorrection"
		break
# 		wait
	done

test_pangolinomni2_pangolinsolo123: ## Evaluate PangolinOmni2 model trained on PangolinSolo123 dataset
	ckpt_dir=checkpoints/pangolinomni2.pangolinsolo123_2025-10-27_07-27-47_933otwx4
	ckpt_path="$${ckpt_dir}/best.ckpt"
	tissues=(liver)
	tissues=(heart liver brain testis)
	tissues=(heart liver brain testis "heart,liver,brain,testis")	
	for ((i=0; i<$${#tissues[@]}; i++)) ; do
		conda run --no-capture-output --name $(CONDA_ENV_NAME) \
		python -u run.py stage=eval \
			infer.save_prefix=$${ckpt_dir}/pangolin_test_$${tissues[i]//,/-} \
			infer.save_level=2 infer.eval_dim=null \
			model.type=pangolinomni2 \
			model.state_dict_path=null \
			litrun.resume_from_ckpt="$${ckpt_path}" \
			dataset.type=pangolinsolo \
			dataset.train_path=null \
			dataset.predict_epoch_length=$(predict_len) \
			dataset.predict_path=data/pangolin/dataset_test_1.h5 \
			+dataset.tissue_types=[$${tissues[i]}] \
			+dataset.tissue_embedding_path="data/tissue_avg_pca_embeddings.csv" \
			debug=$(debug)
		wait
	done

test_pangolinorig_pangolin: ## Evaluate Pangolin original model
	# Run Pangolin model (final.modelnum.tissue.epoch) on Pangolin dataset
	#	dataset.predict_path=data/pangolin/dataset_train_all.h5 \
	models=(final.1.3.3 final.2.3.3 final.3.3.3 final.4.3.3 final.5.3.3 final.1.3.3.v2 final.2.3.3.v2 final.3.3.3.v2)
	models+=(final.1.4.3 final.2.4.3 final.3.4.3 final.4.4.3 final.5.4.3 final.1.4.3.v2 final.2.4.3.v2 final.3.4.3.v2)
	models=(final.1.0.3 final.2.0.3 final.3.0.3 final.4.0.3 final.5.0.3 final.1.0.3.v2 final.2.0.3.v2 final.3.0.3.v2)
# 	models=(final.1.1.3 final.2.1.3 final.3.1.3 final.4.1.3 final.5.1.3 final.1.1.3.v2 final.2.1.3.v2 final.3.1.3.v2)
# 	models+=(final.1.2.3 final.2.2.3 final.3.2.3 final.4.2.3 final.5.2.3 final.1.2.3.v2 final.2.2.3.v2 final.3.2.3.v2)	
	for model in $${models[@]} ; do
		conda run --no-capture-output --name $(CONDA_ENV_NAME) \
		python -u run.py stage=eval \
			infer.save_prefix=benchmarks/pangolin_$${model}/pangolin_test \
			infer.save_level=2 infer.eval_dim=-2 \
			model.type=pangolin \
			model.state_dict_path=benchmarks/pangolin_models/$${model} \
			litrun.resume_from_ckpt=null \
			ensemble.enable=false \
			dataset.type=pangolin \
			dataset.predict_path=data/pangolin/dataset_test_1.h5 \
			dataset.predict_epoch_length=$(predict_len) \
			+dataset.tissue_types=[heart,liver,brain,testis] \
			debug=$(debug)
	done

train_pangolinsolo_pangolinsolo123: ## Train PangolinSolo model on PangolinSolo dataset
	conda run --no-capture-output --name $(CONDA_ENV_NAME) \
	python -u run.py stage=train gpus=[$(gpus)] \
		run_name=pangolinsolo.pangolinsolo123 \
		model.type=pangolinsolo \
		model.state_dict_path=null \
		dataset.type=pangolinsolo \
		+dataset.tissue_types=[heart,liver,brain] \
		dataset.train_path=data/pangolin/dataset_train_all.h5 \
		datamodule.train_batch_size=96 \
		datamodule.val_batch_size=128 \
		litrun.resume_from_ckpt=null \
		debug=$(debug)
