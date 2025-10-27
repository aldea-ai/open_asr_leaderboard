## Benchmarking API Endpoints

### Requirements

- **Python**: 3.11.x
- **PyTorch**: 2.7.1
- **TorchCodec**: 0.3.0

### About `run_eval.py`

`run_eval.py` contains the Python logic for benchmarking each of several API endpoints.
`run_eval.py` accepts core arguments `--model_name`, `--dataset`, which specify the model and dataset to benchmark.
Use this to benchmark one specific model with one specific dataset.
It also accepts optional arguments `--max_workers`, `--use_url`, `--max_samples`, which specify the maximum number of workers to use, whether to use URL-based audio fetching instead of local copies of datasets, and the maximum number of samples to benchmark.
`--dataset_path` will likely never change, and `--split` remains "test" for all datasets, other than `librispeech`, which has a "test.clean" and "test.other" split.

`run_eval.py` is also modified to allow for resuming from where a previous `run_eval.py` call with the same model and dataset left off.
To resume, simply run the same `run_eval.py` call with the same model and dataset.


### Example: Benchmark a single model
```bash
python3.11 run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="gigaspeech" \
        --split="test" \
        --model_name openai/whisper-1 \
        --max_workers 50 \
        --use_url
```

### Batch benchmarking with `run_api.sh`
`run_api.sh` is a bash script that calls `run_eval.py` for each model in `MODEL_IDs` with each of the 8 datasets.
Models benchmarked via `run_api.sh` will benchmark against all 8 datasets.
Feel free to isolate models in `MODEL_IDs` in `run_api.sh` to benchmark specific models against all datasets.


### Example: Run all benchmarks
```bash
nohup bash run_api.sh > run_api.log 2>&1 &
```


### Provider-specific examples

Below are ready-to-run examples for running a single dataset vs all datasets with specific providers.

#### Aldea (single public endpoint) – one dataset
Uses the default cloud endpoint unless overridden. Ensure your API key is set if required.

```bash
export ALDEA_API_KEY=YOUR_TOKEN   # optional if your endpoint requires it

python3.11 run_eval.py \
  --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
  --dataset "gigaspeech" \
  --split "test" \
  --model_name "aldea/default" \
  --max_workers 20
```

To force a specific single endpoint, either set `ALDEA_API_URL` or pass `--aldea_endpoints` with one value:

```bash
export ALDEA_API_URL="https://api.aldea.ai/asr/transcribe"
# or
python3.11 run_eval.py \
  --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
  --dataset "gigaspeech" \
  --split "test" \
  --model_name "aldea/default" \
  --max_workers 20 \
  --aldea_endpoints "https://api.aldea.ai/asr/transcribe"
```

#### Aldea (multiple local endpoints) – one dataset
Round-robins across local ports to increase throughput. You can supply endpoints via env or the CLI flag (space- or comma-separated). Hosts may be `host:port` or full URLs; the `/asr/transcribe` path is added automatically if missing.

```bash
export ALDEA_ENDPOINTS="127.0.0.1:8800 127.0.0.1:8824 127.0.0.1:8848 127.0.0.1:8872 127.0.0.1:8896 127.0.0.1:8920 127.0.0.1:8944 127.0.0.1:8801 127.0.0.1:8825 127.0.0.1:8849 127.0.0.1:8873 127.0.0.1:8897 127.0.0.1:8921 127.0.0.1:8945 127.0.0.1:8802 127.0.0.1:8826 127.0.0.1:8850 127.0.0.1:8874 127.0.0.1:8898 127.0.0.1:8922 127.0.0.1:8946"

python3.11 run_eval.py \
  --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
  --dataset "gigaspeech" \
  --split "test" \
  --model_name "aldea/default" \
  --max_workers 160

# Alternatively (explicit flag):
# --aldea_endpoints "$ALDEA_ENDPOINTS"
```

#### ElevenLabs – one dataset

```bash
export ELEVENLABS_API_KEY=YOUR_TOKEN

python3.11 run_eval.py \
  --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
  --dataset "gigaspeech" \
  --split "test" \
  --model_name "elevenlabs/scribe_v1" \
  --max_workers 5
```

#### Aldea (single public endpoint) – all datasets
Use the batch script with only Aldea enabled. The script automatically forwards `ALDEA_ENDPOINTS` to `run_eval.py` when set, and uses higher concurrency for `aldea/*`.

```bash
# In run_api.sh, set
#   MODEL_IDs=(
#       "aldea/default"
#   )

cd api
nohup bash run_api.sh > run_api.log 2>&1 &
```

#### Aldea (multiple local endpoints) – all datasets

```bash
export ALDEA_ENDPOINTS="127.0.0.1:8800 127.0.0.1:8824 127.0.0.1:8848 127.0.0.1:8872 127.0.0.1:8896 127.0.0.1:8920 127.0.0.1:8944 127.0.0.1:8801 127.0.0.1:8825 127.0.0.1:8849 127.0.0.1:8873 127.0.0.1:8897 127.0.0.1:8921 127.0.0.1:8945 127.0.0.1:8802 127.0.0.1:8826 127.0.0.1:8850 127.0.0.1:8874 127.0.0.1:8898 127.0.0.1:8922 127.0.0.1:8946"

cd api
nohup bash run_api.sh > run_api.log 2>&1 &
```

#### ElevenLabs – all datasets

```bash
# In run_api.sh, set
#   MODEL_IDs=(
#       "elevenlabs/scribe_v1"
#   )

export ELEVENLABS_API_KEY=YOUR_TOKEN

cd api
nohup bash run_api.sh > run_api.log 2>&1 &
```


### Resuming
The `run_eval.py` logic has been modified to allow for resuming from where a previous `run_eval.py` call with the same model and dataset left off.
To resume, simply run the same `run_eval.py` call with the same model and dataset.

### Scoring results
`run_eval.py` outputs a JSONL file to `./results/MODEL_DATASET_SPLIT.jsonl`.
To score the results for a specific model for all associated JSONL files in `./results`, run the following command:
```bash
python3.11 -c "
import sys
sys.path.append('../normalizer')
import eval_utils
eval_utils.score_results('./results', 'elevenlabs/scribe_v1')
"
```

Results for each dataset and composite results will be printed.

### Example output
```text
Filtering models by id: elevenlabs/scribe_v1
********************************************************************************
Results per dataset:
********************************************************************************
elevenlabs/scribe_v1 | hf-audio-esb-datasets-test-only-sorted_ami_test: WER = 19.01 %, RTFx = 1.60
elevenlabs/scribe_v1 | hf-audio-esb-datasets-test-only-sorted_earnings22_test: WER = 11.61 %, RTFx = 2.39
elevenlabs/scribe_v1 | hf-audio-esb-datasets-test-only-sorted_gigaspeech_test: WER = 10.75 %, RTFx = 3.64
elevenlabs/scribe_v1 | hf-audio-esb-datasets-test-only-sorted_librispeech_test.clea: WER = 1.83 %, RTFx = 4.08
elevenlabs/scribe_v1 | hf-audio-esb-datasets-test-only-sorted_librispeech_test.other: WER = 3.67 %, RTFx = 3.52
elevenlabs/scribe_v1 | hf-audio-esb-datasets-test-only-sorted_spgispeech_test: WER = 3.65 %, RTFx = 8.55
elevenlabs/scribe_v1 | hf-audio-esb-datasets-test-only-sorted_tedlium_test: WER = 3.99 %, RTFx = 5.15
elevenlabs/scribe_v1 | hf-audio-esb-datasets-test-only-sorted_voxpopuli_test: WER = 9.34 %, RTFx = 5.14
********************************************************************************
Composite Results:
********************************************************************************
elevenlabs/scribe_v1: WER = 7.98 %
elevenlabs/scribe_v1: RTFx = 3.39
********************************************************************************
```