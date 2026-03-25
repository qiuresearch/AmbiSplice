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
iend=-1
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
	@echo "  profile=singularity|docker"
	@echo "  use_parabricks_star=true|false"
	@echo "  gpus=N"
	@echo "  cpus=N"
	@echo "  debug=true|false"
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

entex_nextflow_rnaseq_fastq: data_dir=entex/downloads_sample_name
entex_nextflow_rnaseq_fastq: out_dir=entex/RNA-seq_dataset
entex_nextflow_rnaseq_fastq: nextflow_rnaseq_fastq ## EnTEX dataset RNA-seq processing with nextflow

encode_nextflow_rnaseq_fastq: data_dir=encode/downloads_organ
encode_nextflow_rnaseq_fastq: out_dir=encode/RNA-seq_dataset
encode_nextflow_rnaseq_fastq: nextflow_rnaseq_fastq ## Encode dataset RNA-seq processing with nextflow

nextflow_rnaseq_fastq: sbatch_redirect ## RNA-seq processing with nextflow
	@echo "Running nextflow rnaseq pipeline on fastq files..."

	@if ! command -v java &> /dev/null ; then module load jdk ; fi
	@echo "java path: $$(which java)"
	@if ! command -v nextflow &> /dev/null ; then
		echo "Nextflow is not available. Please install it first."
		exit 1
	fi
	@echo "nextflow path: $$(which nextflow)"
	@if [ "$(profile)" = "docker" ] ; then 
		if ! command -v docker &> /dev/null ; then module load docker ; fi
		echo "docker path: $$(which docker)"
	elif [ "$(profile)" = "singularity" ] ; then
		if ! command -v singularity &> /dev/null ; then module load singularity ; fi
		echo "singularity path: $$(which singularity)"
	else
		echo "Unknown profile: $(profile)"
		exit 1
	fi

	nf_files=($$(ls $(data_dir)/*_RNA-seq_fastq_nextflow.csv))
	if [ $(iend) -lt 0 ] || [ $(iend) -gt $${#nf_files[@]} ] ; then iend=$${#nf_files[@]} ; else iend=$(iend) ; fi

	fasta_path=$$(realpath nextflow_refdata/GRCh38.primary_assembly.genome.fa)
	gtf_path=$$(realpath nextflow_refdata/gencode.v49.primary_assembly.annotation.gtf)
	transcript_fasta_path=$$(realpath nextflow_refdata/genome.transcripts.fa)
	gene_bed_path=$$(realpath nextflow_refdata/gencode.v49.primary_assembly.annotation.filtered.bed)
	rsem_index_path=$$(realpath nextflow_refdata/rsem)
	star_index_path=$$(realpath nextflow_refdata/index/star)
	salmon_index_path=$$(realpath nextflow_refdata/index/salmon)

	echo "Using profile: $(profile); use_parabricks_star: $(use_parabricks_star); cpus: $(cpus); gpus: $(gpus); debug: $(debug)"
	echo data_dir: $(data_dir), out_dir: $(out_dir)
	echo nf_files: $${nf_files[@]}
	echo "Processing files from index $(istart) to $${iend} (total: $${#nf_files[@]})"
	echo "Reference data files: "
	echo "               fasta: $${fasta_path}"
	echo "                 gtf: $${gtf_path}"
	echo "    transcript_fasta: $${transcript_fasta_path}"
	echo "            gene_bed: $${gene_bed_path}"
	echo "          rsem_index: $${rsem_index_path}"
	echo "          star_index: $${star_index_path}"
	echo "        salmon_index: $${salmon_index_path}"

	data_dir=$$(realpath $(data_dir))
	for ((i=$(istart); i<=$${iend}; i++)) ; do

		nf_file=$$(realpath $${nf_files[i]})
		tissue_type=$$(basename -s _RNA-seq_fastq_nextflow.csv $${nf_files[i]})
		tissue_dir=$(out_dir)/$${tissue_type}
		mkdir -p $${tissue_dir}

		echo "nextflow file: $${nf_files[i]}; tissue: $${tissue_type}"
		pushd $${tissue_dir} > /dev/null

		ln -s $$(realpath --relative-to=. -- $${nf_file}) ./
		ln -s $$(realpath --relative-to=. -- $${data_dir}/$${tissue_type}) $${tissue_type}
		
		nextflow run nf-core/rnaseq -r 3.23.0 \
			-resume \
			-profile $(profile) \
			-work-dir $${HOME}/nextflow_cache \
            --max_cpus $(cpus) \
			--input $${nf_file} \
			--outdir ./ \
			--aligner star_salmon \
            --extra_star_align_args "--alignIntronMax 100000 --alignIntronMin 20"
			--use_parabricks_star $(use_parabricks_star) \
			--stringtie_ignore_gtf true \
			--gencode \
			--fasta $${fasta_path} \
			--gtf $${gtf_path} \
			--transcript_fasta $${transcript_fasta_path} \
			--gene_bed $${gene_bed_path} \
			--rsem_index $${rsem_index_path} \
			--star_index $${star_index_path} \
			--salmon_index $${salmon_index_path}
			
# run it once with save_reference and then use the saved gene_bed, transcriptome fasta, star and salmon indices
# 			--fasta GRCh38.primary_assembly.genome.fa.gz \
# 			--gtf   gencode.v49.primary_assembly.annotation.gtf.gz \
# 			--save_reference
			
# --gpu_container_options '--gpus 1'			
# 			--save_align_intermeds \
#           --skip_alignment 
# Default STAR maxIntronMax is 1000000, reduced to 100000 for better performance and less false positives
# Min 20 and Max 100000 are used in the original Spliser article for human (20 and 6000 for Arabidopsis)
# --alignMatesGapMax 1000000 --alignSJoverhangMin 8 --outFilterMismatchNmax 999 --outFilterMultimapNmax 20 --outFilterType BySJout --outFilterMismatchNoverLmax 0.1 --clip3pAdapterSeq AAAAAAAA
#           --extra_salmon_quant_args "--noLengthCorrection"
#
# 			--additional_fasta to add spike-in sequences (e.g. ERCC), but conflict with --transcript_fast option
# 			--fasta  Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz \ # for use of ensembl references
# 			--gtf Homo_sapiens.GRCh38.115.gtf.gz \

		wait
		popd > /dev/null
		[ "$(debug)" = "true" ] && break
	done

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
		-gff3 gencode.v49.primary_assembly.annotation.gff3 \
		-strand_opt ' --isStranded -s rf' \
		-save_dir entex/spliser

	# echo 'Skip processBamFiles...' || \
	ppl_omics.sh spliser_processBamFiles \
		-home_dir entex \
		-bam_name '*.markdup.sorted.bam' \
		-gff3 gencode.v49.primary_assembly.annotation.gff3 \
		-strand_opt ' --isStranded -s rf' \
		-intron_tsv entex/spliser.introns.tsv \
		-ncpus 4
