.ONESHELL:
SHELL := /bin/bash
.DEFAULT_GOAL := help
# .PHONY: all help clean test_pangolin test_human pangolin_test
# This tells make to always run the recipe for the requested targets,
# regardless of whether a file with that name exists, thus no needs for .PHONY declaration.
MAKEFLAGS += --always-make

# Use ENV_VAR if set, otherwise default to "default_value"

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
	nf_files=($$(ls $${data_dir}/*_RNA-seq_fastq_nextflow.csv))

# 			--additional_fasta to add spike-in sequences (e.g. ERCC), but conflict with --transcript_fast option
# 			--fasta  Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz \
# 			--gtf Homo_sapiens.GRCh38.115.gtf.gz \

# run it once with save_reference and then use the saved gene_bed, transcriptome fasta, star and salmon indices

	for ((i=1; i<$${#nf_files[@]}; i++)) ; do
		tissue_type=$$(basename -s _RNA-seq_fastq_nextflow.csv $${nf_files[i]})
		echo "nextflow file: $${nf_files[i]}; tissue: $${tissue_type}"
		nextflow run nf-core/rnaseq -r 3.23.0 \
			-profile docker \
			--input $${nf_files[i]} \
			--outdir entex/RNA-seq_dataset/$${tissue_type} \
			--aligner star_salmon \
			--stringtie_ignore_gtf true \
			--save_reference \
			--gencode \
			--fasta nextflow_refdata/GRCh38.primary_assembly.genome.fa \
			--gtf nextflow_refdata/gencode.v49.primary_assembly.annotation.gtf \
			--transcript_fasta nextflow_refdata/genome.transcripts.fa \
			--gene_bed nextflow_refdata/gencode.v49.primary_assembly.annotation.filtered.bed \
			--rsem_index nextflow_refdata/rsem \
			--star_index nextflow_refdata/index/star \
			--salmon_index nextflow_refdata/index/salmon

# 			--gtf gencode.v49.primary_assembly.annotation.gtf.gz \
# 			--fasta GRCh38.primary_assembly.genome.fa.gz \
#           --max_cpus \

# 			--save_align_intermeds \
#           --skip_alignment 
#           -resume
# --extra_star_align_args "--alignIntronMax 1000000 --alignIntronMin 20 --alignMatesGapMax 1000000 --alignSJoverhangMin 8 --outFilterMismatchNmax 999 --outFilterMultimapNmax 20 --outFilterType BySJout --outFilterMismatchNoverLmax 0.1 --clip3pAdapterSeq AAAAAAAA"
# --extra_salmon_quant_args "--noLengthCorrection"
		break
# 		wait
	done
