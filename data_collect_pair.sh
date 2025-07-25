#!/bin/bash
set -x
runs=("")
output_dir=""
file_limit=500000000
compress_factor=5

mkdir -p ${output_dir}
rm -rf tmp
sample_spots=100000
true_file_limit=$(($compress_factor * $file_limit))
for run in "${runs[@]}"; do
    mkdir -p tmp
    read -a acc_array <<< "$run"
    fastq-dump -N 1 -X ${sample_spots} --split-3 -O tmp ${acc_array[0]}
    sample_bytes=$(du -b tmp/${acc_array[0]}_1.fastq | awk '{sum += $1} END {print sum}')
    total_spots=$(($true_file_limit * $sample_spots / $sample_bytes))
    
    run_name="${run// /_}"
    touch tmp/${run_name}_1_raw.fastq
    touch tmp/${run_name}_2_raw.fastq
    total_bytes=0
    for acc in "${acc_array[@]}"; do
        touch tmp/${acc}_1.fastq
        touch tmp/${acc}_2.fastq
        fastq-dump -N 1 -X ${total_spots} --split-3 -O tmp ${acc} &
        DUMP_PID=$!

        set +x
        while kill -0 $DUMP_PID 2>/dev/null; do
            bytes=$(du -b tmp/${acc}_1.fastq 2>/dev/null | awk '{sum += $1} END {print sum}')
            curr_bytes=$(($total_bytes + $bytes))
            percent=$((100 * $curr_bytes / $true_file_limit))
            echo "[`date +%T`] Estimated progress: $percent% ($curr_bytes / $true_file_limit Bytes)"
            sleep 10
        done
        set -x
        total_bytes=$(($total_bytes + $bytes))

        lines=$(cat tmp/${acc}_1.fastq | wc -l)
        spots=$((lines / 4))
        total_spots=$(($total_spots - $spots))

        cat tmp/${acc}_1.fastq >> tmp/${run_name}_1_raw.fastq
        cat tmp/${acc}_2.fastq >> tmp/${run_name}_2_raw.fastq
    done

    fastp -i tmp/${run_name}_1_raw.fastq -o tmp/${run_name}_1_trim.fq -I tmp/${run_name}_2_raw.fastq -O tmp/${run_name}_2_trim.fq -j tmp/fastp.json -h tmp/fastp.html
    salmon quant -i salmon_index -l A -1 tmp/${run_name}_1_trim.fq -2 tmp/${run_name}_2_trim.fq -o tmp --validateMappings
    python tx2gene.py -p tmp/quant.sf -o ${output_dir}/${run_name}.csv
    rm -rf tmp
done