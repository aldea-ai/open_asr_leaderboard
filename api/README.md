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
export ALDEA_API_URL="https://api.aldea.ai/transcribe"
# or
python3.11 run_eval.py \
  --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
  --dataset "gigaspeech" \
  --split "test" \
  --model_name "aldea/default" \
  --max_workers 20 \
  --aldea_endpoints "https://api.aldea.ai/transcribe"
```

#### Aldea (multiple local endpoints) – one dataset
Round-robins across local ports to increase throughput. You can supply endpoints via env or the CLI flag (space- or comma-separated). Hosts may be `host:port` or full URLs; the `/transcribe` path is added automatically if missing.

```bash
export ALDEA_ENDPOINTS="http://stt-api-test.aldea.ai:8800 http://stt-api-test.aldea.ai:8824 http://stt-api-test.aldea.ai:8848 http://stt-api-test.aldea.ai:8872 http://stt-api-test.aldea.ai:8896 http://stt-api-test.aldea.ai:8920 http://stt-api-test.aldea.ai:8944 http://stt-api-test.aldea.ai:8801 http://stt-api-test.aldea.ai:8825 http://stt-api-test.aldea.ai:8849 http://stt-api-test.aldea.ai:8873 http://stt-api-test.aldea.ai:8897 http://stt-api-test.aldea.ai:8921 http://stt-api-test.aldea.ai:8945 http://stt-api-test.aldea.ai:8802 http://stt-api-test.aldea.ai:8826 http://stt-api-test.aldea.ai:8850 http://stt-api-test.aldea.ai:8874 http://stt-api-test.aldea.ai:8898 http://stt-api-test.aldea.ai:8922 http://stt-api-test.aldea.ai:8946"

python3.11 run_eval.py \
  --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
  --dataset "gigaspeech" \
  --split "test" \
  --model_name "aldea/default" \
  --max_workers 160

# Alternatively (explicit flag):
# --aldea_endpoints "$ALDEA_ENDPOINTS"
```

### Rate limiting for Aldea

This runner enforces a per-endpoint rate limit for Aldea `/transcribe` requests.

- Default pacing is 0.5 requests per second (one request every 2 seconds) per endpoint.
- Override via the `ALDEA_RPS` environment variable:

```bash
export ALDEA_RPS=0.5   # default; 1 request every 2 seconds per endpoint
```

If you supply multiple endpoints via `ALDEA_ENDPOINTS`, throughput scales roughly with the number of endpoints, since each endpoint has its own limiter.

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
export ALDEA_ENDPOINTS="http://stt-api-test.aldea.ai:8800 http://stt-api-test.aldea.ai:8824 http://stt-api-test.aldea.ai:8848 http://stt-api-test.aldea.ai:8872 http://stt-api-test.aldea.ai:8896 http://stt-api-test.aldea.ai:8920 http://stt-api-test.aldea.ai:8944 http://stt-api-test.aldea.ai:8801 http://stt-api-test.aldea.ai:8825 http://stt-api-test.aldea.ai:8849 http://stt-api-test.aldea.ai:8873 http://stt-api-test.aldea.ai:8897 http://stt-api-test.aldea.ai:8921 http://stt-api-test.aldea.ai:8945 http://stt-api-test.aldea.ai:8802 http://stt-api-test.aldea.ai:8826 http://stt-api-test.aldea.ai:8850 http://stt-api-test.aldea.ai:8874 http://stt-api-test.aldea.ai:8898 http://stt-api-test.aldea.ai:8922 http://stt-api-test.aldea.ai:8946"

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

### Dataset sizes (samples and hours)
You can quickly get sample counts and approximate total audio hours per dataset/split using the helper script:

```bash
# counts + hours for all default datasets/splits
python3.11 get_dataset_sizes.py

# just counts (faster)
python3.11 get_dataset_sizes.py --no_duration

# only specific datasets
python3.11 get_dataset_sizes.py --datasets librispeech gigaspeech

# add/override splits
python3.11 get_dataset_sizes.py --include_splits librispeech:test.clean librispeech:test.other

# to avoid rate limits (recommended)
export HF_TOKEN=... && python3.11 get_dataset_sizes.py
```

Faster or more resilient options:

```bash
# estimate total hours using only the first N rows (fastest)
python3.11 get_dataset_sizes.py --estimate_duration --sample_rows 2000

# add retry/backoff and gentle pacing between pages
python3.11 get_dataset_sizes.py \
  --batch 200 \
  --progress_every 10 \
  --max_retries 10 \
  --initial_backoff 2 \
  --sleep_between_pages 0.1
```

Sample results:
bash```
bash-3.2$ python get_dataset_sizes.py --no_duration
dataset	split	samples
[ami:test] querying size...
ami	test	12643
[earnings22:test] querying size...
earnings22	test	2741
[gigaspeech:test] querying size...
gigaspeech	test	19931
[librispeech:test.clean] querying size...
librispeech	test.clean	5559
[librispeech:test.other] querying size...
librispeech	test.other	5559
[spgispeech:test] querying size...
spgispeech	test	39341
[tedlium:test] querying size...
tedlium	test	1155
[voxpopuli:test] querying size...
voxpopuli	test	1842
```

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