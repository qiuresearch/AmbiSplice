
# source ${HOME}/programs/spliser/spliser_venv/bin/activate

tissue="Heart"
gff3="${HOME}/database/genomes/gencode.v49lift37.annotation.gff3"
brew_omics.py get_bam_stats -refbed ~/database/genomes/gencode.v49lift37.annotation.bed \
		-bamfiles bam_files/*${tissue}*.sorted.sorted.bam \
		-save_prefix ${tissue,,}_bam_stats

exit 0
save_prefix="spliser_${tissue,,}"
bai_files=($(ls bam_files/*.${tissue}.*sorted.sorted.bai))
echo "Found a total of ${#bai_files[@]} bai files"

joined_bams=$(printf '%s,' "${bai_files[@]}")
joined_bams=${joined_bams//.bai,/.bam,}
echo "joined bams: ${joined_bams}"

# spliser preCombineIntrons -o spliser_heart -A ${gff3} --isStranded -s rf -L "${joined_bams%,}"

bam_prefixes=${joined_bams//.bam,/\ }
bam_prefixes=(echo ${bam_prefixes//bam_files\//})
echo "Found a total of ${#bam_prefixes[@]} bam files: ${bam_prefixes[@]}"

# printf '%s\n' ${bam_prefixes[@]} | xargs -n 1 -I prefix -P 20 spliser process -o ${save_prefix}/prefix -A ${gff3} -t gene \
	# -I ${save_prefix}.introns.tsv --isStranded -s rf -B bam_files/prefix.bam

spliser_tsvs=($(ls ${save_prefix}/*.SpliSER.tsv))
echo "Found a total of ${#spliser_tsvs[@]} spliser tsv files"

sample_tsv="${save_prefix}.samples.tsv"
echo -n "" > ${sample_tsv}
for tsv in ${spliser_tsvs[@]}
do
	sample_id=$(basename ${tsv%.SpliSER.tsv})
	echo -e "${sample_id}\t${tsv}\tbam_files/${sample_id}.bam" >> ${sample_tsv}
done

# spliser combine -S ${sample_tsv} -o ${save_prefix} --isStranded -s rf

spliser output -S ${sample_tsv} -C ${save_prefix}.combined.tsv -t DiffSpliSER --minReads 10 -o ${save_prefix}.

