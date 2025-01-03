#!/bin/bash

MODEL_ID=meta-llama/Meta-Llama-3-70B-Instruct


red='\033[31m'
green='\033[32m'
blue='\033[34m'
reset='\033[0m'


echo "\n******** 🔥🔥 ${red}Make sure that vLLM runs${reset} ${green} $MODEL_ID${reset} 🔥🔥 ********\n"
sleep 2


ID=0
END_INDEX=2


FUNC_WORDS_LIST=(
  "_"
)

cd ../evaluation
# find ./ONLINE_TARGETS -type d -name "out_*" -exec rm -rf {} +
for TGT_LANG in de; do
    for WAIT_K in 2; do
        for PROMPT_ID in 1; do
            for FUNC_WORDS in "${FUNC_WORDS_LIST[@]}"; do
                echo "ID: $ID"
                simuleval \
                    --source SOURCES/src_ted_new_tst_100.$TGT_LANG.txt \
                    --target OFFLINE_TARGETS/tgt_ted_new_tst_100.$TGT_LANG \
                    --background BACKGROUND_INFO/bgd_ted_new_tst_100.$TGT_LANG \
                    --agent t2tt_agent.py \
                    --k $WAIT_K \
                    --output ONLINE_TARGETS/out_${0}_${ID} \
                    --start-index 0 \
                    --end-index $END_INDEX \
                    --verbose \
                    --dir en-$TGT_LANG \
                    --use_api \
                    --latency-metrics LAAL AL AP DAL \
                    --quality-metrics BLEU CHRF \
                    --model_id $MODEL_ID \
                    --prompt_id $PROMPT_ID \
                    --func_wrds $FUNC_WORDS \
                    --priming
                ID=$((ID+1))

                simuleval \
                    --source SOURCES/src_ted_new_tst_100.$TGT_LANG.txt \
                    --target OFFLINE_TARGETS/tgt_ted_new_tst_100.$TGT_LANG \
                    --agent t2tt_agent.py \
                    --k $WAIT_K \
                    --output ONLINE_TARGETS/out_${0}_${ID} \
                    --start-index 0 \
                    --end-index $END_INDEX \
                    --verbose \
                    --dir en-$TGT_LANG \
                    --use_api \
                    --latency-metrics LAAL AL AP DAL \
                    --quality-metrics BLEU CHRF \
                    --model_id $MODEL_ID \
                    --prompt_id $PROMPT_ID \
                    --func_wrds $FUNC_WORDS \
                    --priming
                ID=$((ID+1))
            done
        done
    done
done