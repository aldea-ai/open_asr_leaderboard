#!/bin/bash

export PYTHONPATH="/Users/kevin/Documents/aldea/open_asr_leaderboard"
echo "PYTHONPATH: $PYTHONPATH"

# Load specific API keys from .env file
if [ -f ../.env ]; then
    echo "Loading API keys from ../.env file..."
    export OPENAI_API_KEY=$(grep "^OPENAI_API_KEY=" ../.env | sed 's/.*"\([^"]*\)".*/\1/')
    export ASSEMBLYAI_API_KEY=$(grep "^ASSEMBLYAI_API_KEY=" ../.env | sed 's/.*"\([^"]*\)".*/\1/')
    export ELEVENLABS_API_KEY=$(grep "^ELEVENLABS_API_KEY=" ../.env | sed 's/.*"\([^"]*\)".*/\1/')
    export REVAI_API_KEY=$(grep "^REVAI_API_KEY=" ../.env | sed 's/.*"\([^"]*\)".*/\1/')
    export DEEPGRAM_API_KEY=$(grep "^DEEPGRAM_API_KEY=" ../.env | sed 's/.*"\([^"]*\)".*/\1/')
    export GROQ_API_KEY=$(grep "^GROQ_API_KEY=" ../.env | sed 's/.*"\([^"]*\)".*/\1/')
    export HF_TOKEN=$(grep "^HF_TOKEN=" ../.env | sed 's/.*"\([^"]*\)".*/\1/')
    export ALDEA_API_KEY=$(grep "^ALDEA_API_KEY=" ../.env | sed 's/.*"\([^"]*\)".*/\1/')
    export SPEECHMATICS_API_KEY=$(grep "^SPEECHMATICS_API_KEY=" ../.env | sed 's/.*"\([^"]*\)".*/\1/')

else
    echo "Warning: .env file not found. Please create a .env file with your API keys."
    exit 1
fi

# export MODEL_ID="assembly/slam-1"
# export MAX_WORKERS=180

MODEL_IDs=(
    # "openai/whisper-1"
    # "groq/whisper-large-v3"
    # "deepgram/nova-3-general"
    # "openai/gpt-4o-transcribe"
    # "openai/gpt-4o-mini-transcribe"
    # "elevenlabs/scribe_v1"
    # "assembly/slam-1"
    "assembly/universal"
    # "revai/machine" # please use --use_url=True
    # "revai/fusion" # please use --use_url=True
    # "speechmatics-batch/enhanced"  # Batch (async) API
    # "speechmatics-rt/enhanced"     # Real-time (WebSocket) API
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
        echo 60  
    elif [[ $model_id == deepgram/* ]]; then
        echo 50
    elif [[ $model_id == speechmatics-rt/* ]]; then
        echo 15  # Conservative limit for WebSocket connections
    elif [[ $model_id == speechmatics-batch/* ]]; then
        echo 30  # Batch can handle more
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
        --max_workers ${MAX_WORKERS} 
        # --use_url
        # --max_samples ${MAX_SAMPLES}


    echo "Running test.clean for ${MODEL_ID} on earnings22"
    python3.11 run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="earnings22" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_workers ${MAX_WORKERS} 
        # --max_samples ${MAX_SAMPLES}

    echo "Running test.clean for ${MODEL_ID} on gigaspeech"
    python3.11 run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="gigaspeech" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_workers ${MAX_WORKERS} 
        # --max_samples ${MAX_SAMPLES}

    echo "Running test.clean for ${MODEL_ID} on librispeech"
    python3.11 run_eval.py \
        --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
        --dataset "librispeech" \
        --split "test.clean" \
        --model_name ${MODEL_ID} \
        --max_workers ${MAX_WORKERS} 
        # --max_samples ${MAX_SAMPLES}

    echo "Running test.other for ${MODEL_ID} on librispeech"
    python3.11 run_eval.py \
        --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
        --dataset "librispeech" \
        --split "test.other" \
        --model_name ${MODEL_ID} \
        --max_workers ${MAX_WORKERS} 
        # --max_samples ${MAX_SAMPLES}
    
    echo "Running test.other for ${MODEL_ID} on spgispeech"
    python3.11 run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="spgispeech" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_workers ${MAX_WORKERS} 
        # --max_samples ${MAX_SAMPLES}

    echo "Running test.other for ${MODEL_ID} on tedlium"
    python3.11 run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="tedlium" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_workers ${MAX_WORKERS} \
        # --max_samples ${MAX_SAMPLES}

    echo "Running test.other for ${MODEL_ID} on voxpopuli"
    python3.11 run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="voxpopuli" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_workers ${MAX_WORKERS} 
        # --max_samples ${MAX_SAMPLES}

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python3.11 -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
