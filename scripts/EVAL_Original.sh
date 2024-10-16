#!/bin/bash

ASR_MODEL=distil-large-v3
echo "ASR model: $ASR_MODEL"

echo "\n******** 🔥🔥 ${red}Make sure that vLLM runs${reset} ${green} $MODEL_ID${reset} 🔥🔥 ********\n"
sleep 2

FUNC_WORDS_LIST=(
  "_"
)

END_INDEX=70
END_INDEX_2023=102
ID=0

cd ../evaluation
# find ./ONLINE_TARGETS -type d -name "out_*" -exec rm -rf {} +

# TED-TST-2023 Testings
for TGT_LANG in de; do
    for FUNC_WORDS in "${FUNC_WORDS_LIST[@]}"; do
        echo $FUNC_WORDS
        for K in 1; do
            simuleval \
                --source SOURCES/src_ted_new_tst_100.$TGT_LANG \
                --target OFFLINE_TARGETS/tgt_ted_new_tst_100.$TGT_LANG \
                --background BACKGROUND_INFO/bgd_ted_new_tst_100_brief.$TGT_LANG \
                --agent s2tt_agent.py \
                --k $K \
                --output ONLINE_TARGETS/out_${0}_TED2023_${TGT_LANG} \
                --start-index 0 \
                --end-index $END_INDEX_2023 \
                --verbose \
                --dir en-$TGT_LANG \
                --use_api \
                --latency-metrics LAAL AL AP DAL \
                --quality-metrics BLEU CHRF \
                --model_id meta-llama/Meta-Llama-3-70B-Instruct \
                --source-segment-size 200 \
                --use_asr_api \
                --asr_model_size $ASR_MODEL \
                --prompt_id 1 \
                --func_wrds $FUNC_WORDS \
                --priming
            ID=$((ID+1))
        done
    done
done

# TED-TST-2024 Testings
for TGT_LANG in de fr es ru it; do
    for FUNC_WORDS in "${FUNC_WORDS_LIST[@]}"; do
        echo $FUNC_WORDS
        for K in 1; do
            for MIN_READ_TIME in 1.4; do
                simuleval \
                    --source SOURCES/ted_tst_2024 \
                    --background BACKGROUND_INFO/ted_tst_2024.txt \
                    --target OFFLINE_TARGETS/ted_tst_2024_$TGT_LANG.txt \
                    --agent s2tt_agent.py \
                    --k $K \
                    --output ONLINE_TARGETS/out_${0}_TED2024_${TGT_LANG}_Background_Priming \
                    --start-index 0 \
                    --end-index $END_INDEX_2023 \
                    --verbose \
                    --dir en-$TGT_LANG \
                    --use_api \
                    --latency-metrics LAAL AL AP DAL \
                    --quality-metrics BLEU CHRF \
                    --model_id meta-llama/Meta-Llama-3-70B-Instruct \
                    --source-segment-size 200 \
                    --min_read_time $MIN_READ_TIME \
                    --use_asr_api \
                    --asr_model_size $ASR_MODEL \
                    --min_lag_words 1 \
                    --prompt_id 1 \
                    --func_wrds $FUNC_WORDS \
                    --priming
                ID=$((ID+1))

                simuleval \
                    --source SOURCES/ted_tst_2024 \
                    --target OFFLINE_TARGETS/ted_tst_2024_$TGT_LANG.txt \
                    --agent s2tt_agent.py \
                    --k $K \
                    --output ONLINE_TARGETS/out_${0}_TED2024_${TGT_LANG}_Priming \
                    --start-index 0 \
                    --end-index $END_INDEX_2023 \
                    --verbose \
                    --dir en-$TGT_LANG \
                    --use_api \
                    --latency-metrics LAAL AL AP DAL \
                    --quality-metrics BLEU CHRF \
                    --model_id meta-llama/Meta-Llama-3-70B-Instruct \
                    --source-segment-size 200 \
                    --min_read_time $MIN_READ_TIME \
                    --use_asr_api \
                    --asr_model_size $ASR_MODEL \
                    --min_lag_words 1 \
                    --prompt_id 1 \
                    --func_wrds $FUNC_WORDS \
                    --priming
                ID=$((ID+1))

                simuleval \
                    --source SOURCES/ted_tst_2024 \
                    --background BACKGROUND_INFO/ted_tst_2024.txt \
                    --target OFFLINE_TARGETS/ted_tst_2024_$TGT_LANG.txt \
                    --agent s2tt_agent.py \
                    --k $K \
                    --output ONLINE_TARGETS/out_${0}_TED2024_${TGT_LANG}_Background \
                    --start-index 0 \
                    --end-index $END_INDEX_2023 \
                    --verbose \
                    --dir en-$TGT_LANG \
                    --use_api \
                    --latency-metrics LAAL AL AP DAL \
                    --quality-metrics BLEU CHRF \
                    --model_id meta-llama/Meta-Llama-3-70B-Instruct \
                    --source-segment-size 200 \
                    --min_read_time $MIN_READ_TIME \
                    --use_asr_api \
                    --asr_model_size $ASR_MODEL \
                    --min_lag_words 1 \
                    --prompt_id 1 \
                    --func_wrds $FUNC_WORDS \
                ID=$((ID+1))
            done
        done
    done
done