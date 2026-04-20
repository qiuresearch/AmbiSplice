.ONESHELL:
SHELL := /bin/bash
.DEFAULT_GOAL := help
# This tells make to always run the recipe for the requested targets,
# regardless of whether a file with that name exists, thus no needs for .PHONY declaration.
MAKEFLAGS += --always-make
# .PHONY: all help clean test_pangolin test_human pangolin_test

# Use ENV_VAR if set, otherwise default to "default_value"

debug=false
gpus=0
cpus=$(shell echo "scale=0; m=$$(nproc)/2; if(m<12) m else 12" | bc)

istart=0
iend=9999
profile=singularity
use_parabricks_star=false

sbatch=false
time=7-00:00:00
partition=cpu
mem=96G


help: ## Display this help message
	@echo "Usage: make <target>"
	@echo
	@echo "Available options:"
	@echo "  profile=singularity|docker (default: $(profile))"
	@echo "  use_parabricks_star=true|false (default: $(use_parabricks_star))"
	@echo "  gpus=N (default: $(gpus))"
	@echo "  cpus=N (default: $(cpus))"
	@echo "  debug=true|false (default: $(debug))"
	@echo
	@echo "Available targets:"
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-42s\033[0m %s\n", $$1, $$2}'

sbatch_redirect: ## Redirect the target action to sbatch (sbatch=true)
	@if [ -n "$${SLURM_JOB_ID}" ] ; then exit 0 ; fi
	@GOALOPT="istart=$(istart) iend=$(iend) profile=$(profile) cpus=$(cpus) gpus=$(gpus) debug=$(debug)"
	@if [ "$(sbatch)" = "true" ] ; then
		sbatch_file=sbatch
		sbatch_cmds=(echo Starting...)
		for goal in $(MAKECMDGOALS) ; do
			sbatch_file=$${sbatch_file}_$${goal}_$(istart)_$(iend)
			sbatch_cmds+=(\; make $${goal} $${GOALOPT})
		done
		sbatch_brew.sh -p $(partition) -t $(time) -ncpus $(cpus) -o "$${sbatch_file}.sh" "$${sbatch_cmds[*]}"
		if command -v sbatch &> /dev/null ; then
			echo "Submitting via sbatch ... (ignore the 'sbatch: error message)"
			sbatch "$${sbatch_file}.sh"
		else
			echo "sbatch command not found. Please submit $${sbatch_file}.sh manually"
		fi
		exit 1
	fi

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
	wget -c --no-proxy -L https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.primary_assembly.annotation.gff3.gz

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

nextflow_rnaseq_test: nextflow_check ## run nextflow test on rnaseq data
	nextflow run nf-core/rnaseq -r 3.23.0 -profile test_full,$(profile) --outdir test --max_cpus 6

nextflow_rnaseq_batch: sbatch_redirect
	ppl_nextflow.sh nextflow_rnaseq \
		-profile $(profile) \
		-use_parabricks_star $(use_parabricks_star) \
		-cpus $(cpus) \
		-gpus $(gpus) \
		-data_dir $(data_dir) \
		-out_dir $(out_dir) \
		-url_list $(url_list) \
		-istart $(istart) \
		-iend $(iend) \
		-debug $(debug)

entex_nextflow_rnaseq_batch: data_dir=entex/downloads_sample_name
entex_nextflow_rnaseq_batch: out_dir=entex/RNA-seq_dataset
entex_nextflow_rnaseq_batch: url_list=$(data_dir)_cloud.urls
entex_nextflow_rnaseq_batch: nextflow_rnaseq_batch ## EnTEX dataset RNA-seq processing with nextflow

encode_nextflow_rnaseq_batch: data_dir=encode/downloads_organ
encode_nextflow_rnaseq_batch: out_dir=encode/RNA-seq_dataset
encode_nextflow_rnaseq_batch: url_list=$(data_dir)_cloud.urls
encode_nextflow_rnaseq_batch: nextflow_rnaseq_batch ## Encode dataset RNA-seq processing with nextflow

nextflow_atacseq_batch: sbatch_redirect
	ppl_nextflow.sh nextflow_atacseq \
		-profile $(profile) \
		-use_parabricks_star $(use_parabricks_star) \
		-cpus $(cpus) \
		-gpus $(gpus) \
		-data_dir $(data_dir) \
		-out_dir $(out_dir) \
		-url_list $(url_list) \
		-istart $(istart) \
		-iend $(iend) \
		-debug $(debug)

entex_nextflow_atacseq_batch: data_dir=entex/downloads_biosample_name
entex_nextflow_atacseq_batch: out_dir=entex/ATAC-seq_dataset
entex_nextflow_atacseq_batch: url_list=$(data_dir)_cloud.urls
entex_nextflow_atacseq_batch: nextflow_atacseq_batch ## EnTEX dataset ATAC-seq processing with nextflow

entex_spliser_rnaseq: ## EnTEX dataset RNA-seq processing with SpliSer
	echo 'Skip collectBamStats...' || \
	ppl_omics.sh spliser_collectBamStats \
		-home_dir entex \
		-glob_file multiqc_fail_strand_check_table.txt \
		-save_dir entex/spliser

	echo 'Skip preCombineIntrons...' || \
	ppl_omics.sh splier_preCombineIntrons \
		-home_dir entex \
		-bam_name '*.markdup.sorted.bam' \
		-gff3 spliser_refdata/gencode.v49.primary_assembly.annotation.gff3 \
		-strand_opt ' --isStranded -s rf' \
		-save_dir entex/spliser

	# echo 'Skip processBamFiles...' || \
	ppl_omics.sh spliser_processBamFiles \
		-home_dir entex \
		-bam_name '*.markdup.sorted.bam' \
		-gff3 gencode.v49.primary_assembly.annotation.gff3 \
		-strand_opt ' --isStranded -s rf' \
		-intron_tsv entex/spliser.introns.tsv \
		-ncpus $(cpus)
