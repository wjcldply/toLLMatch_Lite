#!/bin/bash

ASR_MODEL=small
echo "ASR model: $ASR_MODEL"

echo -e "\n******** 🔥🔥 ${red}Make sure that vLLM runs${reset} ${green} $MODEL_ID${reset} 🔥🔥 ********\n"
sleep 2

FUNC_WORDS_LIST=(
  "_"
)

END_INDEX=70
ID=0

cd ../evaluation

# find ./ONLINE_TARGETS -type d -name "out_*" -exec rm -rf {} +
for TGT_LANG in de fr es ru it; do
    for FUNC_WORDS in "${FUNC_WORDS_LIST[@]}"; do
        printf "Processing functional words: %s\r" "$FUNC_WORDS"
        for K in 1; do
            for MIN_READ_TIME in 1.4; do
                printf "Running simuleval for Target Lang: %s | Func Words: %s | ID: %d\r" "$TGT_LANG" "$FUNC_WORDS" "$ID"
                simuleval \
                    --source SOURCES/ted_tst_2024 \
                    --background BACKGROUND_INFO/ted_tst_2024.txt \
                    --target OFFLINE_TARGETS/ted_tst_2024_$TGT_LANG.txt \
                    --agent s2tt_agent_updated.py \
                    --k $K \
                    --output ONLINE_TARGETS/updated_${0}_${ID}_${TGT_LANG}_w_Background \
                    --start-index 0 \
                    --end-index $END_INDEX \
                    --verbose \
                    --dir en-$TGT_LANG \
                    --use_api \
                    --latency-metrics LAAL AL AP DAL \
                    --quality-metrics BLEU CHRF \
                    --model_id meta-llama/Meta-Llama-3-8B-Instruct \
                    --source-segment-size 200 \
                    --min_read_time $MIN_READ_TIME \
                    --use_asr_api \
                    --asr_model_size $ASR_MODEL \
                    --min_lag_words 1 \
                    --prompt_id 1 \
                    --func_wrds $FUNC_WORDS \
                    --priming
                ID=$((ID+1))
            done
        done
    done
    printf "Completed for Target Lang: %s\n" "$TGT_LANG"
done