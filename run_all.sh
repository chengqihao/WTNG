#!/bin/bash
# run.sh - 自动化运行脚本

# 设置严格模式，遇到错误退出
set -e
set -o pipefail


cd ./build/test

WARMUP_RUNS="${WARMUP_RUNS:-2}"
MEASURE_RUNS="${MEASURE_RUNS:-5}"
CPU_NODE="${CPU_NODE:-0}"
MEM_NODE="${MEM_NODE:-0}"
CPU_CORE="${CPU_CORE:-0}"

run_bound() {
    numactl --cpunodebind="${CPU_NODE}" --membind="${MEM_NODE}" \
        taskset -c "${CPU_CORE}" "$@"
}

average_search_outputs() {
    awk '
    function is_avg_line(line) {
        return line ~ /^search time:/ ||
               line ~ /^average stop depth:/ ||
               line ~ /^DistCount:/ ||
               line ~ /^HopCount:/ ||
               line ~ /^[[:space:]]*[0-9]+[[:space:]]+NN accuracy:/ ||
               line ~ /^[[:space:]]*VmPeak:/ ||
               line ~ /^[[:space:]]*VmHWM:/
    }

    function split_metric(line, colon_pos, rest, start_pos, len) {
        colon_pos = index(line, ":")
        if (colon_pos == 0) {
            return 0
        }

        rest = substr(line, colon_pos + 1)
        if (!match(rest, /[-+]?[0-9]+([.][0-9]*)?([eE][-+]?[0-9]+)?|[-+]?[.][0-9]+([eE][-+]?[0-9]+)?/)) {
            return 0
        }

        start_pos = colon_pos + RSTART
        len = RLENGTH
        metric_prefix = substr(line, 1, start_pos - 1)
        metric_value = substr(line, start_pos, len) + 0
        metric_suffix = substr(line, start_pos + len)
        return 1
    }

    FNR == 1 {
        run++
        line_no = 0
    }

    {
        line_no++
        if (run == 1) {
            first_line[line_no] = $0
            max_line = line_no
        }

        if (is_avg_line($0) && split_metric($0)) {
            if (run == 1) {
                prefix[line_no] = metric_prefix
                suffix[line_no] = metric_suffix
                should_avg[line_no] = 1
            }
            sum[line_no] += metric_value
            count[line_no]++
        }
    }

    END {
        for (i = 1; i <= max_line; i++) {
            if (should_avg[i] && count[i] > 0) {
                printf "%s%.10g%s\n", prefix[i], sum[i] / count[i], suffix[i]
            } else {
                print first_line[i]
            }
        }
    }
    ' "$@"
}

run_search() {
    local output_mode="$1"
    local outfile="$2"
    local tmpdir
    local run
    local output_file
    local measured_outputs=()
    shift 2

    tmpdir="$(mktemp -d)"

    for ((run = 1; run <= WARMUP_RUNS; run++)); do
        echo "[warmup ${run}/${WARMUP_RUNS}] $*" >&2
        if ! run_bound "$@" > /dev/null; then
            rm -rf "$tmpdir"
            return 1
        fi
    done

    for ((run = 1; run <= MEASURE_RUNS; run++)); do
        echo "[measure ${run}/${MEASURE_RUNS}] $*" >&2
        output_file="${tmpdir}/run_${run}.log"
        measured_outputs+=("$output_file")
        if ! run_bound "$@" > "$output_file"; then
            rm -rf "$tmpdir"
            return 1
        fi
    done

    case "$output_mode" in
        append)
            average_search_outputs "${measured_outputs[@]}" | tee -a "$outfile"
            ;;
        write)
            average_search_outputs "${measured_outputs[@]}" | tee "$outfile"
            ;;
        *)
            echo "Unknown output mode: $output_mode" >&2
            rm -rf "$tmpdir"
            return 1
            ;;
    esac

    rm -rf "$tmpdir"
}

Data1="mugen"

echo "===================="
echo "Running experiment..."
echo "Dataset: $Data1"
echo "===================="


mkdir -p ../../plot_figure/mugen

# cp -av "../../plot_figure"/*.py "../../plot_figure/${Data1}"/

# alpha_set_1=(0.7 0.1 0.1 0.05 0.45 0.45 0.3)
# alpha_set_2=(0.1 0.7 0.1 0.45 0.05 0.45 0.3)

./main wtng mugen 0.7 0.1 1 1 1 build | tee -a  "../../plot_figure/${Data1}/wtng_build.txt"

OUTFILE="../../plot_figure/${Data1}/wtng_alpha_0.3_0.3.txt"
echo "Running with alpha=0.3,0.3, saving to $OUTFILE"
run_search write "$OUTFILE" ./main wtng mugen 0.3 0.3 1 1 1 search

./main baseline1 mugen 0.7 0.1 1 1 1 build | tee -a  "../../plot_figure/${Data1}/bs1_build.txt"

OUTFILE="../../plot_figure/${Data1}/bs1_alpha_0.3_0.3.txt"
echo "Running with alpha=0.3,0.3, saving to $OUTFILE"
run_search write "$OUTFILE" ./main baseline1 mugen 0.3 0.3 1 1 1 search

# for i in "${!alpha_set_1[@]}"; 
# do
#     alpha1="${alpha_set_1[i]}"
#     alpha2="${alpha_set_2[i]}"
#     OUTFILE="../../plot_figure/${Data1}/wtng_alpha_${alpha1}_${alpha2}.txt"
#     echo "Running with alpha=${alpha1},${alpha2}, saving to $OUTFILE"
#     run_search write "$OUTFILE" ./main wtng mugen $alpha1 $alpha2 1 1 1 search
# done

# ./main baseline1 mugen 0.7 0.1 1 1 1 build | tee -a  "../../plot_figure/${Data1}/bs1_build.txt"

# for i in "${!alpha_set_1[@]}"; 
# do
#     alpha1="${alpha_set_1[i]}"
#     alpha2="${alpha_set_2[i]}"
#     OUTFILE="../../plot_figure/${Data1}/bs1_alpha_${alpha1}_${alpha2}.txt"
#     echo "Running with alpha=${alpha1},${alpha2}, saving to $OUTFILE"
#     run_search append "$OUTFILE" ./main baseline1 mugen $alpha1 $alpha2 1 1 1 search
# done

# ./main baseline2 mugen 0.7 0.1 1 1 1  build | tee -a  "../../plot_figure/${Data1}/bs2_build.txt"

# for i in "${!alpha_set_1[@]}"; 
# do
#     alpha1="${alpha_set_1[i]}"
#     alpha2="${alpha_set_2[i]}"
#     OUTFILE="../../plot_figure/${Data1}/bs2_alpha_${alpha1}_${alpha2}.txt"
#     echo "Running with alpha=${alpha1},${alpha2}, saving to $OUTFILE"
#     run_search append "$OUTFILE" ./main baseline2 mugen $alpha1 $alpha2 1 1 1 search
# done

# ./main vbase mugen 0.7 0.1 1 1 1  build | tee -a  "../../plot_figure/${Data1}/vbase_build.txt"

# for i in "${!alpha_set_1[@]}"; 
# do
#     alpha1="${alpha_set_1[i]}"
#     alpha2="${alpha_set_2[i]}"
#     OUTFILE="../../plot_figure/${Data1}/vbase_alpha_${alpha1}_${alpha2}.txt"
#     echo "Running with alpha=${alpha1},${alpha2}, saving to $OUTFILE"
#     run_search append "$OUTFILE" ./main vbase mugen $alpha1 $alpha2 1 1 1 search
# done


echo "Experiment finished. All logs saved."

# Data2="openimage"

# echo "===================="
# echo "Running experiment..."
# echo "Dataset: $Data2"
# echo "===================="

# mkdir -p ../../plot_figure/openimage

# cp -av "../../plot_figure"/*.py "../../plot_figure/${Data2}"/

# alpha_set_1=(0.7 0.1 0.1 0.05 0.45 0.45 0.3)
# alpha_set_2=(0.1 0.7 0.1 0.45 0.05 0.45 0.3)

# ./main wtng openimage 0.7 0.1 1 1 1 build | tee -a  "../../plot_figure/${Data2}/wtng_build.txt"

# for i in "${!alpha_set_1[@]}"; 
# do
#     alpha1="${alpha_set_1[i]}"
#     alpha2="${alpha_set_2[i]}"
#     OUTFILE="../../plot_figure/${Data2}/wtng_alpha_${alpha1}_${alpha2}.txt"
#     echo "Running with alpha=${alpha1},${alpha2}, saving to $OUTFILE"
#     run_search write "$OUTFILE" ./main wtng openimage $alpha1 $alpha2 1 1 1 search
# done

# ./main baseline1 openimage 0.7 0.1 1 1 1 build | tee -a  "../../plot_figure/${Data2}/bs1_build.txt"

# for i in "${!alpha_set_1[@]}"; 
# do
#     alpha1="${alpha_set_1[i]}"
#     alpha2="${alpha_set_2[i]}"
#     OUTFILE="../../plot_figure/${Data2}/bs1_alpha_${alpha1}_${alpha2}.txt"
#     echo "Running with alpha=${alpha1},${alpha2}, saving to $OUTFILE"
#     run_search append "$OUTFILE" ./main baseline1 openimage $alpha1 $alpha2 1 1 1 search
# done

# ./main baseline2 openimage 0.7 0.1 1 1 1  build | tee -a  "../../plot_figure/${Data2}/bs2_build.txt"

# for i in "${!alpha_set_1[@]}"; 
# do
#     alpha1="${alpha_set_1[i]}"
#     alpha2="${alpha_set_2[i]}"
#     OUTFILE="../../plot_figure/${Data2}/bs2_alpha_${alpha1}_${alpha2}.txt"
#     echo "Running with alpha=${alpha1},${alpha2}, saving to $OUTFILE"
#     run_search append "$OUTFILE" ./main baseline2 openimage $alpha1 $alpha2 1 1 1 search
# done

# ./main vbase openimage 0.7 0.1 1 1 1  build | tee -a  "../../plot_figure/${Data2}/vbase_build.txt"

# for i in "${!alpha_set_1[@]}"; 
# do
#     alpha1="${alpha_set_1[i]}"
#     alpha2="${alpha_set_2[i]}"
#     OUTFILE="../../plot_figure/${Data2}/vbase_alpha_${alpha1}_${alpha2}.txt"
#     echo "Running with alpha=${alpha1},${alpha2}, saving to $OUTFILE"
#     run_search append "$OUTFILE" ./main vbase openimage $alpha1 $alpha2 1 1 1 search
# done

# echo "Experiment finished. All logs saved."




# Data3="imagenet"

# echo "===================="
# echo "Running experiment..."
# echo "Dataset: $Data3"
# echo "===================="

# mkdir -p ../../plot_figure/imagenet

# cp -av "../../plot_figure"/*.py "../../plot_figure/${Data3}"/

# alpha_set_1=(0.7 0.1 0.1 0.05 0.45 0.45 0.3)
# alpha_set_2=(0.1 0.7 0.1 0.45 0.05 0.45 0.3)

# ./main wtng imagenet 0.7 0.1 1 1 1 build | tee -a  "../../plot_figure/${Data3}/wtng_build.txt"

# for i in "${!alpha_set_1[@]}"; 
# do
#     alpha1="${alpha_set_1[i]}"
#     alpha2="${alpha_set_2[i]}"
#     OUTFILE="../../plot_figure/${Data3}/wtng_alpha_${alpha1}_${alpha2}.txt"
#     echo "Running with alpha=${alpha1},${alpha2}, saving to $OUTFILE"
#     run_search write "$OUTFILE" ./main wtng imagenet $alpha1 $alpha2 1 1 1 search
# done

# ./main baseline1 imagenet 0.7 0.1 1 1 1 build | tee -a  "../../plot_figure/${Data3}/bs1_build.txt"

# for i in "${!alpha_set_1[@]}"; 
# do
#     alpha1="${alpha_set_1[i]}"
#     alpha2="${alpha_set_2[i]}"
#     OUTFILE="../../plot_figure/${Data3}/bs1_alpha_${alpha1}_${alpha2}.txt"
#     echo "Running with alpha=${alpha1},${alpha2}, saving to $OUTFILE"
#     run_search append "$OUTFILE" ./main baseline1 imagenet $alpha1 $alpha2 1 1 1 search
# done

# ./main baseline2 imagenet 0.7 0.1 1 1 1  build | tee -a  "../../plot_figure/${Data3}/bs2_build.txt"

# for i in "${!alpha_set_1[@]}"; 
# do
#     alpha1="${alpha_set_1[i]}"
#     alpha2="${alpha_set_2[i]}"
#     OUTFILE="../../plot_figure/${Data3}/bs2_alpha_${alpha1}_${alpha2}.txt"
#     echo "Running with alpha=${alpha1},${alpha2}, saving to $OUTFILE"
#     run_search append "$OUTFILE" ./main baseline2 imagenet $alpha1 $alpha2 1 1 1 search
# done

# ./main vbase imagenet 0.7 0.1 1 1 1  build | tee -a  "../../plot_figure/${Data3}/vbase_build.txt"

# for i in "${!alpha_set_1[@]}"; 
# do
#     alpha1="${alpha_set_1[i]}"
#     alpha2="${alpha_set_2[i]}"
#     OUTFILE="../../plot_figure/${Data3}/vbase_alpha_${alpha1}_${alpha2}.txt"
#     echo "Running with alpha=${alpha1},${alpha2}, saving to $OUTFILE"
#     run_search append "$OUTFILE" ./main vbase imagenet $alpha1 $alpha2 1 1 1 search
# done





# echo "Experiment finished. All logs saved."

# Data4="cc3m"

# echo "===================="
# echo "Running experiment..."
# echo "Dataset: Data4"
# echo "===================="

# mkdir -p ../../plot_figure/cc3m

# cp -av "../../plot_figure"/*.py "../../plot_figure/${Data4}"/

# alpha_set_1=(0.7 0.1 0.1 0.05 0.45 0.45 0.3)
# alpha_set_2=(0.1 0.7 0.1 0.45 0.05 0.45 0.3)

# ./main wtng cc3m 0.7 0.1 1 1 1 build | tee -a  "../../plot_figure/${Data4}/wtng_build.txt"

# for i in "${!alpha_set_1[@]}"; 
# do
#     alpha1="${alpha_set_1[i]}"
#     alpha2="${alpha_set_2[i]}"
#     OUTFILE="../../plot_figure/${Data4}/wtng_alpha_${alpha1}_${alpha2}.txt"
#     echo "Running with alpha=${alpha1},${alpha2}, saving to $OUTFILE"
#     run_search write "$OUTFILE" ./main wtng cc3m $alpha1 $alpha2 1 1 1 search
# done

# ./main baseline1 cc3m 0.7 0.1 1 1 1 build | tee -a  "../../plot_figure/${Data4}/bs1_build.txt"

# for i in "${!alpha_set_1[@]}"; 
# do
#     alpha1="${alpha_set_1[i]}"
#     alpha2="${alpha_set_2[i]}"
#     OUTFILE="../../plot_figure/${Data4}/bs1_alpha_${alpha1}_${alpha2}.txt"
#     echo "Running with alpha=${alpha1},${alpha2}, saving to $OUTFILE"
#     run_search append "$OUTFILE" ./main baseline1 cc3m $alpha1 $alpha2 1 1 1 search
# done

# ./main baseline2 cc3m 0.7 0.1 1 1 1  build | tee -a  "../../plot_figure/${Data4}/bs2_build.txt"

# for i in "${!alpha_set_1[@]}"; 
# do
#     alpha1="${alpha_set_1[i]}"
#     alpha2="${alpha_set_2[i]}"
#     OUTFILE="../../plot_figure/${Data4}/bs2_alpha_${alpha1}_${alpha2}.txt"
#     echo "Running with alpha=${alpha1},${alpha2}, saving to $OUTFILE"
#     run_search append "$OUTFILE" ./main baseline2 cc3m $alpha1 $alpha2 1 1 1 search
# done

# ./main vbase cc3m 0.7 0.1 1 1 1  build | tee -a  "../../plot_figure/${Data4}/vbase_build.txt"

# for i in "${!alpha_set_1[@]}"; 
# do
#     alpha1="${alpha_set_1[i]}"
#     alpha2="${alpha_set_2[i]}"
#     OUTFILE="../../plot_figure/${Data4}/vbase_alpha_${alpha1}_${alpha2}.txt"
#     echo "Running with alpha=${alpha1},${alpha2}, saving to $OUTFILE"
#     run_search append "$OUTFILE" ./main vbase cc3m $alpha1 $alpha2 1 1 1 search
# done

# echo "Experiment finished. All logs saved."


Data5="msmarco"

echo "===================="
echo "Running experiment..."
echo "Dataset: $Data5"
echo "===================="

mkdir -p ../../plot_figure/msmarco

cp -av "../../plot_figure"/*.py "../../plot_figure/${Data5}"/

alpha_set_1=(0.7 0.1 0.1 0.05 0.45 0.45 0.3)
alpha_set_2=(0.1 0.7 0.1 0.45 0.05 0.45 0.3)

./main wtng msmarco 0.7 0.1 1 1 1 build | tee -a  "../../plot_figure/${Data5}/wtng_build.txt"

for i in "${!alpha_set_1[@]}"; 
do
    alpha1="${alpha_set_1[i]}"
    alpha2="${alpha_set_2[i]}"
    OUTFILE="../../plot_figure/${Data5}/wtng_alpha_${alpha1}_${alpha2}.txt"
    echo "Running with alpha=${alpha1},${alpha2}, saving to $OUTFILE"
    run_search write "$OUTFILE" ./main wtng msmarco $alpha1 $alpha2 1 1 1 search
done

./main baseline1 msmarco 0.7 0.1 1 1 1 build | tee -a  "../../plot_figure/${Data5}/bs1_build.txt"

for i in "${!alpha_set_1[@]}"; 
do
    alpha1="${alpha_set_1[i]}"
    alpha2="${alpha_set_2[i]}"
    OUTFILE="../../plot_figure/${Data5}/bs1_alpha_${alpha1}_${alpha2}.txt"
    echo "Running with alpha=${alpha1},${alpha2}, saving to $OUTFILE"
    run_search append "$OUTFILE" ./main baseline1 msmarco $alpha1 $alpha2 1 1 1 search
done

./main baseline2 msmarco 0.7 0.1 1 1 1  build | tee -a  "../../plot_figure/${Data5}/bs2_build.txt"

for i in "${!alpha_set_1[@]}"; 
do
    alpha1="${alpha_set_1[i]}"
    alpha2="${alpha_set_2[i]}"
    OUTFILE="../../plot_figure/${Data5}/bs2_alpha_${alpha1}_${alpha2}.txt"
    echo "Running with alpha=${alpha1},${alpha2}, saving to $OUTFILE"
    run_search append "$OUTFILE" ./main baseline2 msmarco $alpha1 $alpha2 1 1 1 search
done

./main vbase msmarco 0.7 0.1 1 1 1  build | tee -a  "../../plot_figure/${Data5}/vbase_build.txt"

for i in "${!alpha_set_1[@]}"; 
do
    alpha1="${alpha_set_1[i]}"
    alpha2="${alpha_set_2[i]}"
    OUTFILE="../../plot_figure/${Data5}/vbase_alpha_${alpha1}_${alpha2}.txt"
    echo "Running with alpha=${alpha1},${alpha2}, saving to $OUTFILE"
    run_search append "$OUTFILE" ./main vbase msmarco $alpha1 $alpha2 1 1 1 search
done

echo "Experiment finished. All logs saved."