#!/bin/bash

export PYTHONPATH="/Users/kevin/Documents/aldea/open_asr_leaderboard"
echo "PYTHONPATH: $PYTHONPATH"

export OPENAI_API_KEY="your_api_key"
export ASSEMBLYAI_API_KEY="21b8985515d84ebe96f3b09c0353d84e" #loaded from .env
export ELEVENLABS_API_KEY="your_api_key"
export REVAI_API_KEY="your_api_key"
export DEEPGRAM_API_KEY="a0ebb9c2307b9c5f23e10a187168feb351d6dc81" #loaded from .env

MODEL_IDs=(
    "deepgram/nova-3-general"
    # "openai/gpt-4o-transcribe"
    # "openai/gpt-4o-mini-transcribe"
    # "openai/whisper-1"
    # "assembly/best"
    # "elevenlabs/scribe_v1"
    # "revai/machine" # please use --use_url=True
    # "revai/fusion" # please use --use_url=True
    # "speechmatics/enhanced"
)

MAX_WORKERS=10
MAX_SAMPLES=10

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}
    echo "Running test.clean for ${MODEL_ID} on ami"
    python3.11 run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="ami" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_workers ${MAX_WORKERS} \
        # --max_samples ${MAX_SAMPLES}


    echo "Running test.clean for ${MODEL_ID} on earnings22"
    python3.11 run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="earnings22" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_workers ${MAX_WORKERS} \
        # --max_samples ${MAX_SAMPLES}

    echo "Running test.clean for ${MODEL_ID} on gigaspeech"
    python3.11 run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="gigaspeech" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_workers ${MAX_WORKERS} \
        # --max_samples ${MAX_SAMPLES}

    echo "Running test.clean for ${MODEL_ID} on librispeech"
    python3.11 run_eval.py \
        --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
        --dataset "librispeech" \
        --split "test.clean" \
        --model_name ${MODEL_ID} \
        --max_workers ${MAX_WORKERS} \
        # --max_samples ${MAX_SAMPLES}

    echo "Running test.other for ${MODEL_ID} on librispeech"
    python3.11 run_eval.py \
        --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
        --dataset "librispeech" \
        --split "test.other" \
        --model_name ${MODEL_ID} \
        --max_workers ${MAX_WORKERS} \
        # --max_samples ${MAX_SAMPLES}
    
    echo "Running test.other for ${MODEL_ID} on spgispeech"
    python3.11 run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="spgispeech" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_workers ${MAX_WORKERS} \
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
        --max_workers ${MAX_WORKERS} \
        # --max_samples ${MAX_SAMPLES}

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python3.11 -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
