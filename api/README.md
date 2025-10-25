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
```

### Batch benchmarking with `run_api.sh`
`run_api.sh` is a bash script that calls `run_eval.py` for each model in `MODEL_IDs` with each of the 8 datasets.
Models benchmarked via `run_api.sh` will benchmark against all 8 datasets.
Feel free to isolate models in `MODEL_IDs` in `run_api.sh` to benchmark specific models against all datasets.


### Example: Run all benchmarks
```bash
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