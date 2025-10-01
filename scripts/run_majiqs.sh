#!/bin/bash

# conda deactivate || echo "Ignore this error!"
source ${HOME}/programs/majiq/bin/activate

cd E-MTAB-6814

tissues=(Heart Liver Kidney Testis)
tissues=(Kidney)
tissues=(Testis)

for tissue in "${tissues[@]}" ; do

    tsv_file="majiq_${tissue,,}.tsv"

    ppl_omics.sh majiq_experiment_tsv "${tsv_file}" *${tissue}*.bam
    ppl_omics.sh majiq_build ${tsv_file}

    build_dir="${tsv_file%.tsv}_build"
    ppl_omics.sh majiq_psicov "${build_dir}"

    psicov_dir="${build_dir%_build}_psicov"
    ppl_omics.sh majiq_moccasin "${psicov_dir}"

    mocca_dir="${psicov_dir%_psicov}_mocca"
    ppl_omics.sh majiq_psi "${mocca_dir}"
done
