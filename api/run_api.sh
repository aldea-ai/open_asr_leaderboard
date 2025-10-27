#!/bin/bash

export PYTHONPATH="/Users/kevin/Documents/aldea/open_asr_leaderboard"
echo "PYTHONPATH: $PYTHONPATH"

export OPENAI_API_KEY=""
export ASSEMBLYAI_API_KEY=""
export ELEVENLABS_API_KEY=""
export REVAI_API_KEY=""
export DEEPGRAM_API_KEY="" #loaded from .env
export GROQ_API_KEY=""
# export HF_TOKEN="" # old
export HF_TOKEN="" # new
export ALDEA_API_KEY=""
# Optional: list of local Aldea endpoints (comma or space separated)
# e.g., "127.0.0.1:8800 127.0.0.1:8824 ..."
export ALDEA_ENDPOINTS=""

# export MODEL_ID="assembly/slam-1"
# export MAX_WORKERS=180

MODEL_IDs=(
    "openai/whisper-1"
    # "groq/whisper-large-v3"
    # "deepgram/nova-3-general"
    # "openai/gpt-4o-transcribe"
    # "openai/gpt-4o-mini-transcribe"
    # "elevenlabs/scribe_v1"
    # "assembly/slam-1"
    # "revai/machine" # please use --use_url=True
    # "revai/fusion" # please use --use_url=True
    # "speechmatics/enhanced"
    # "aldea/default"
)

MAX_SAMPLES=400

# Function to get appropriate max_workers based on model provider
get_max_workers() {
    local model_id=$1
    if [[ $model_id == groq/* ]]; then
        echo 8
    elif [[ $model_id == elevenlabs/* ]]; then
        echo 5  
    elif [[ $model_id == assembly/* ]]; then
        echo 200  
    elif [[ $model_id == deepgram/* ]]; then
        echo 50  
    elif [[ $model_id == aldea/* ]]; then
        # Favor high concurrency when using many local endpoints
        echo 180
    else
        echo 20  
    fi
}

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}
    MAX_WORKERS=$(get_max_workers $MODEL_ID)
    echo "Using MAX_WORKERS=$MAX_WORKERS for model: $MODEL_ID"
    echo "Running test.clean for ${MODEL_ID} on ami"
    python3.11 run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="ami" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_workers ${MAX_WORKERS} \
        ${ALDEA_ENDPOINTS:+--aldea_endpoints "$ALDEA_ENDPOINTS"}
        # --use_url
        # --max_samples ${MAX_SAMPLES}


    echo "Running test.clean for ${MODEL_ID} on earnings22"
    python3.11 run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="earnings22" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_workers ${MAX_WORKERS} \
        ${ALDEA_ENDPOINTS:+--aldea_endpoints "$ALDEA_ENDPOINTS"}
        # --max_samples ${MAX_SAMPLES}

    echo "Running test.clean for ${MODEL_ID} on gigaspeech"
    python3.11 run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="gigaspeech" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_workers ${MAX_WORKERS} \
        ${ALDEA_ENDPOINTS:+--aldea_endpoints "$ALDEA_ENDPOINTS"}
        # --max_samples ${MAX_SAMPLES}

    echo "Running test.clean for ${MODEL_ID} on librispeech"
    python3.11 run_eval.py \
        --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
        --dataset "librispeech" \
        --split "test.clean" \
        --model_name ${MODEL_ID} \
        --max_workers ${MAX_WORKERS} \
        ${ALDEA_ENDPOINTS:+--aldea_endpoints "$ALDEA_ENDPOINTS"}
        # --max_samples ${MAX_SAMPLES}

    echo "Running test.other for ${MODEL_ID} on librispeech"
    python3.11 run_eval.py \
        --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
        --dataset "librispeech" \
        --split "test.other" \
        --model_name ${MODEL_ID} \
        --max_workers ${MAX_WORKERS} \
        ${ALDEA_ENDPOINTS:+--aldea_endpoints "$ALDEA_ENDPOINTS"}
        # --max_samples ${MAX_SAMPLES}
    
    echo "Running test.other for ${MODEL_ID} on spgispeech"
    python3.11 run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="spgispeech" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_workers ${MAX_WORKERS} \
        ${ALDEA_ENDPOINTS:+--aldea_endpoints "$ALDEA_ENDPOINTS"}
        # --max_samples ${MAX_SAMPLES}

    echo "Running test.other for ${MODEL_ID} on tedlium"
    python3.11 run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="tedlium" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_workers ${MAX_WORKERS} \
        ${ALDEA_ENDPOINTS:+--aldea_endpoints "$ALDEA_ENDPOINTS"} \
        # --max_samples ${MAX_SAMPLES}

    echo "Running test.other for ${MODEL_ID} on voxpopuli"
    python3.11 run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="voxpopuli" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_workers ${MAX_WORKERS} \
        ${ALDEA_ENDPOINTS:+--aldea_endpoints "$ALDEA_ENDPOINTS"}
        # --max_samples ${MAX_SAMPLES}

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python3.11 -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
