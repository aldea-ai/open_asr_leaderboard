import argparse
from typing import Optional
import datasets
import evaluate
import soundfile as sf
import tempfile
import time
import os
import requests
import itertools
from tqdm import tqdm
from dotenv import load_dotenv
from io import BytesIO
import assemblyai as aai
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from rev_ai import apiclient
from rev_ai.models import CustomerUrlData
from normalizer import data_utils
from normalizer import eval_utils
import concurrent.futures
from speechmatics.models import ConnectionSettings, BatchTranscriptionConfig, FetchData
from speechmatics.batch_client import BatchClient
from httpx import HTTPStatusError
from requests_toolbelt import MultipartEncoder
from email.utils import parsedate_to_datetime
import threading
import json

load_dotenv()

# Default connect/read timeouts for all external HTTP calls to avoid hangs
DEFAULT_CONNECT_TIMEOUT_S = 10
DEFAULT_READ_TIMEOUT_S = 300
REQUEST_TIMEOUT = (DEFAULT_CONNECT_TIMEOUT_S, DEFAULT_READ_TIMEOUT_S)


# Global state for Aldea endpoint round-robin
ALDEA_ENDPOINTS = []
_ALDEA_CYCLE = None
_ALDEA_LOCK = threading.Lock()


def _normalize_aldea_endpoint(endpoint: str) -> str:
    e = (endpoint or "").strip()
    if not e:
        return e
    if not (e.startswith("http://") or e.startswith("https://")):
        e = f"http://{e}"
    # Ensure path suffix
    if not e.rstrip("/").endswith("asr/transcribe"):
        e = e.rstrip("/") + "/asr/transcribe"
    return e


def set_aldea_endpoints(endpoints: list[str]) -> None:
    global ALDEA_ENDPOINTS, _ALDEA_CYCLE
    normalized = [_normalize_aldea_endpoint(ep) for ep in endpoints if ep and ep.strip()]
    ALDEA_ENDPOINTS = [ep for ep in normalized if ep]
    if ALDEA_ENDPOINTS:
        # Create a cycle iterator once; guarded by lock for thread-safety on next()
        import itertools as _itertools
        _ALDEA_CYCLE = _itertools.cycle(ALDEA_ENDPOINTS)
    else:
        _ALDEA_CYCLE = None


def next_aldea_endpoint() -> Optional[str]:
    if _ALDEA_CYCLE is None:
        return None
    with _ALDEA_LOCK:
        return next(_ALDEA_CYCLE)


# Simple thread-safe token bucket rate limiter for max throughput without bursts
class TokenBucketRateLimiter:
    def __init__(self, rate_per_minute: int, capacity: Optional[int] = None) -> None:
        self.rate_per_minute = max(1, int(rate_per_minute))
        self.capacity = capacity if capacity is not None else self.rate_per_minute
        self.tokens = float(self.capacity)
        self.refill_rate_per_second = self.rate_per_minute / 60.0
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> None:
        if tokens <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self.last_refill
                if elapsed > 0:
                    self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate_per_second)
                    self.last_refill = now

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return

                needed = tokens - self.tokens
                wait_seconds = needed / self.refill_rate_per_second if self.refill_rate_per_second > 0 else 0.1
            # Sleep outside the lock so other threads can progress/refill
            time.sleep(max(0.0, wait_seconds))


# Global limiter specifically for OpenAI whisper-1 to cap at 500 requests per minute
OPENAI_WHISPER1_RPM = int(os.getenv("OPENAI_WHISPER1_RPM", "500"))
OPENAI_WHISPER1_LIMITER = TokenBucketRateLimiter(rate_per_minute=OPENAI_WHISPER1_RPM)


def fetch_audio_urls(dataset_path, dataset, split, batch_size=100, max_retries=20):
    print("Fetching audio URLs")
    API_URL = "https://datasets-server.huggingface.co/rows"

    size_url = f"https://datasets-server.huggingface.co/size?dataset={dataset_path}&config={dataset}&split={split}"
    print("Size URL:", size_url)
    hf_headers = {}
    if os.environ.get("HF_TOKEN") is not None:
        hf_headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    size_response = requests.get(size_url, headers=hf_headers, timeout=REQUEST_TIMEOUT).json()
    print("Size response:", size_response)
    
    # Check for errors in the response
    if "error" in size_response:
        raise ValueError(f"Dataset API error: {size_response['error']}")
    
    total_rows = size_response["size"]["config"]["num_rows"]
    audio_urls = []
    for offset in tqdm(range(0, total_rows, batch_size), desc="Fetching audio URLs"):
        params = {
            "dataset": dataset_path,
            "config": dataset,
            "split": split,
            "offset": offset,
            "length": min(batch_size, total_rows - offset),
        }

        retries = 0
        while retries <= max_retries:
            try:
                headers = {}
                if os.environ.get("HF_TOKEN") is not None:
                    headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
                else:
                    print("HF_TOKEN not set, might experience rate-limiting.")
                response = requests.get(API_URL, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                data = response.json()
                yield from data["rows"]
                break
            except (requests.exceptions.RequestException, ValueError) as e:
                retries += 1
                print(
                    f"Error fetching data: {e}, retrying ({retries}/{max_retries})..."
                )
                time.sleep(10)
                if retries >= max_retries:
                    raise Exception("Max retries exceeded while fetching data.")


def transcribe_with_retry(
    model_name: str,
    audio_file_path: Optional[str],
    sample: dict,
    max_retries=10,
    use_url=False,
):
    retries = 0
    while retries <= max_retries:
        try:
            PREFIX = "speechmatics/"
            if model_name.startswith(PREFIX):
                api_key = os.getenv("SPEECHMATICS_API_KEY")
                if not api_key:
                    raise ValueError(
                        "SPEECHMATICS_API_KEY environment variable not set"
                    )

                settings = ConnectionSettings(
                    url="https://asr.api.speechmatics.com/v2", auth_token=api_key
                )
                with BatchClient(settings) as client:
                    config = BatchTranscriptionConfig(
                        language="en",
                        enable_entities=True,
                        operating_point=model_name[len(PREFIX) :],
                    )

                    job_id = None
                    audio_url = None
                    try:
                        if use_url:
                            audio_url = sample["row"]["audio"][0]["src"]
                            config.fetch_data = FetchData(url=audio_url)
                            multipart_data = MultipartEncoder(
                                fields={"config": config.as_config().encode("utf-8")}
                            )
                            response = client.send_request(
                                "POST",
                                "jobs",
                                data=multipart_data.to_string(),
                                headers={"Content-Type": multipart_data.content_type},
                            )
                            job_id = response.json()["id"]
                        else:
                            job_id = client.submit_job(audio_file_path, config)

                        transcript = client.wait_for_completion(
                            job_id, transcription_format="txt"
                        )
                        return transcript
                    except HTTPStatusError as e:
                        if e.response.status_code == 401:
                            raise ValueError(
                                "Invalid Speechmatics API credentials"
                            ) from e
                        elif e.response.status_code == 400:
                            raise ValueError(
                                f"Speechmatics API responded with 400 Bad request: {e.response.text}"
                            )
                        raise e
                    except Exception as e:
                        if job_id is not None:
                            status = client.check_job_status(job_id)
                            if (
                                audio_url is not None
                                and "job" in status
                                and "errors" in status["job"]
                                and isinstance(status["job"]["errors"], list)
                                and len(status["job"]["errors"]) > 0
                            ):
                                errors = status["job"]["errors"]
                                if "message" in errors[-1] and "failed to fetch file" in errors[-1]["message"]:
                                    retries = max_retries + 1
                                    raise Exception(f"could not fetch URL {audio_url}, not retrying")

                        raise Exception(
                            f"Speechmatics transcription failed: {str(e)}"
                        ) from e

            elif model_name.startswith("assembly/"):
                aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
                transcriber = aai.Transcriber()
                config = aai.TranscriptionConfig(
                    speech_model=model_name.split("/")[1],
                    language_code="en",
                )
                if use_url:
                    audio_url = sample["row"]["audio"][0]["src"]
                    audio_duration = sample["row"]["audio_length_s"]
                    if audio_duration < 0.160:
                        print(f"Skipping audio duration {audio_duration}s")
                        return "."
                    transcript = transcriber.transcribe(audio_url, config=config)
                else:
                    audio_duration = (
                        len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]
                    )
                    if audio_duration < 0.160:
                        print(f"Skipping audio duration {audio_duration}s")
                        return "."
                    transcript = transcriber.transcribe(audio_file_path, config=config)

                if transcript.status == aai.TranscriptStatus.error:
                    raise Exception(
                        f"AssemblyAI transcription error: {transcript.error}"
                    )
                return transcript.text

            elif model_name.startswith("openai/"):
                client = OpenAI()
                model_id = model_name.split("/")[1]
                if use_url:
                    response = requests.get(sample["row"]["audio"][0]["src"], timeout=REQUEST_TIMEOUT)
                    audio_data = BytesIO(response.content)
                    audio_data.name = "audio.wav"
                    if model_id == "whisper-1":
                        OPENAI_WHISPER1_LIMITER.acquire()
                    result = client.audio.transcriptions.create(
                        model=model_id,
                        file=audio_data,
                    )
                else:
                    with open(audio_file_path, "rb") as audio_file:
                        if model_id == "whisper-1":
                            OPENAI_WHISPER1_LIMITER.acquire()
                        result = client.audio.transcriptions.create(
                            model=model_id,
                            file=audio_file,
                        )
                return result.text.strip()

            elif model_name.startswith("elevenlabs/"):
                client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
                if use_url:
                    response = requests.get(sample["row"]["audio"][0]["src"], timeout=REQUEST_TIMEOUT)
                    audio_data = BytesIO(response.content)
                    transcription = client.speech_to_text.convert(
                        file=audio_data,
                        model_id=model_name.split("/")[1],
                        language_code="eng",
                        tag_audio_events=True,
                    )
                else:
                    with open(audio_file_path, "rb") as audio_file:
                        transcription = client.speech_to_text.convert(
                            file=audio_file,
                            model_id=model_name.split("/")[1],
                            language_code="eng",
                            tag_audio_events=True,
                        )
                return transcription.text

            elif model_name.startswith("revai/"):
                access_token = os.getenv("REVAI_API_KEY")
                client = apiclient.RevAiAPIClient(access_token)

                if use_url:
                    # Submit job with URL for Rev.ai
                    job = client.submit_job_url(
                        transcriber=model_name.split("/")[1],
                        source_config=CustomerUrlData(sample["row"]["audio"][0]["src"]),
                        metadata="benchmarking_job",
                    )
                else:
                    # Submit job with local file
                    job = client.submit_job_local_file(
                        transcriber=model_name.split("/")[1],
                        filename=audio_file_path,
                        metadata="benchmarking_job",
                    )

                # Polling until job is done
                while True:
                    job_details = client.get_job_details(job.id)
                    if job_details.status.name in ["IN_PROGRESS", "TRANSCRIBING"]:
                        time.sleep(0.1)
                        continue
                    elif job_details.status.name == "FAILED":
                        raise Exception("RevAI transcription failed.")
                    elif job_details.status.name == "TRANSCRIBED":
                        break

                transcript_object = client.get_transcript_object(job.id)

                # Combine all words from all monologues
                transcript_text = []
                for monologue in transcript_object.monologues:
                    for element in monologue.elements:
                        transcript_text.append(element.value)

                return "".join(transcript_text) if transcript_text else ""

            elif model_name.startswith("deepgram/"):
                api_key = os.getenv("DEEPGRAM_API_KEY")
                if not api_key:
                    raise ValueError("DEEPGRAM_API_KEY environment variable not set")

                model_id = model_name.split("/")[1]
                headers = {"Authorization": f"Token {api_key}"}
                params = {
                    "model": model_id,
                    "language": "en",
                    "smart_format": "true",
                }

                if use_url:
                    audio_url = sample["row"]["audio"][0]["src"]
                    audio_duration = sample["row"]["audio_length_s"]
                    if audio_duration < 0.160:
                        print(f"Skipping audio duration {audio_duration}s")
                        return "."
                    headers["Content-Type"] = "application/json"
                    response = requests.post(
                        "https://api.deepgram.com/v1/listen",
                        headers=headers,
                        params=params,
                        json={"url": audio_url},
                        timeout=REQUEST_TIMEOUT,
                    )
                else:
                    audio_duration = (
                        len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]
                    )
                    if audio_duration < 0.160:
                        print(f"Skipping audio duration {audio_duration}s")
                        return "."
                    headers["Content-Type"] = "audio/wav"
                    with open(audio_file_path, "rb") as f:
                        response = requests.post(
                            "https://api.deepgram.com/v1/listen",
                            headers=headers,
                            params=params,
                            data=f,
                            timeout=REQUEST_TIMEOUT,
                        )

                response.raise_for_status()
                result = response.json()

                # Robustly extract transcript across API variants
                transcript = ""
                try:
                    alternatives = result["results"]["channels"][0]["alternatives"]
                    if len(alternatives) > 0:
                        alt = alternatives[0]
                        if isinstance(alt, dict):
                            if "transcript" in alt and isinstance(alt["transcript"], str):
                                transcript = alt["transcript"]
                            elif "paragraphs" in alt and isinstance(alt["paragraphs"], dict) and "transcript" in alt["paragraphs"]:
                                transcript = alt["paragraphs"]["transcript"]
                            elif "words" in alt and isinstance(alt["words"], list):
                                transcript = " ".join([w.get("word", "") for w in alt["words"]]).strip()
                except Exception:
                    pass

                if not isinstance(transcript, str):
                    transcript = str(transcript)

                return transcript

            elif model_name.startswith("groq/"):
                
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("GROQ_API_KEY environment variable not set")

                model_id = model_name.split("/")[1]
                headers = {"Authorization": f"Bearer {api_key}"}

                # Check audio duration first (before acquiring lock)
                if use_url:
                    audio_duration = sample["row"]["audio_length_s"]
                else:
                    audio_duration = (
                        len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]
                    )
                
                if audio_duration < 0.160:
                    print(f"Skipping audio duration {audio_duration}s")
                    return "."

                # Rate limiting removed - Groq now allows 300 requests/minute
                # No artificial delays needed
                if use_url:
                    # For URL-based, download the audio first
                    audio_url = sample["row"]["audio"][0]["src"]
                    response = requests.get(audio_url, timeout=REQUEST_TIMEOUT)
                    audio_data = BytesIO(response.content)
                    audio_data.name = "audio.wav"
                    
                    files = {
                        "file": ("audio.wav", audio_data, "audio/wav"),
                        "model": (None, model_id),
                        "response_format": (None, "verbose_json"),
                    }
                    response = requests.post(
                        "https://api.groq.com/openai/v1/audio/transcriptions",
                        headers=headers,
                        files=files,
                        timeout=REQUEST_TIMEOUT,
                    )
                    if response.status_code == 429:
                        # Respect Retry-After if provided by Groq
                        retry_after = response.headers.get("Retry-After") or response.headers.get("retry-after")
                        delay = None
                        if retry_after is not None:
                            try:
                                delay = int(float(retry_after))
                            except ValueError:
                                try:
                                    dt = parsedate_to_datetime(retry_after)
                                    delay = max(1, int((dt - datetime.datetime.now(datetime.timezone.utc)).total_seconds()))
                                except Exception:
                                    delay = None
                        if delay is None:
                            delay = min(60, 2 ** (retries + 1))
                        print(f"Rate limit 429 from Groq. Retrying in {delay}s... (Attempt {retries+1}/{max_retries})")
                        time.sleep(delay)
                        retries += 1
                        continue
                else:
                    with open(audio_file_path, "rb") as f:
                        files = {
                            "file": ("audio.wav", f, "audio/wav"),
                            "model": (None, model_id),
                            "response_format": (None, "verbose_json"),
                        }
                        response = requests.post(
                            "https://api.groq.com/openai/v1/audio/transcriptions",
                            headers=headers,
                            files=files,
                            timeout=REQUEST_TIMEOUT,
                        )
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After") or response.headers.get("retry-after")
                    delay = None
                    if retry_after is not None:
                        try:
                            delay = int(float(retry_after))
                        except ValueError:
                            try:
                                dt = parsedate_to_datetime(retry_after)
                                delay = max(1, int((dt - datetime.datetime.now(datetime.timezone.utc)).total_seconds()))
                            except Exception:
                                delay = None
                    if delay is None:
                        delay = min(60, 2 ** (retries + 1))
                    print(f"Rate limit 429 from Groq. Retrying in {delay}s... (Attempt {retries+1}/{max_retries})")
                    time.sleep(delay)
                    retries += 1
                    continue
                response.raise_for_status()
                result = response.json()

                # Extract transcript from response
                transcript = result.get("text", "")
                
                if not isinstance(transcript, str):
                    transcript = str(transcript)

                return transcript

            elif model_name.startswith("aldea/"):
                # Choose endpoint via round-robin if provided; otherwise fall back to env or default
                api_url = next_aldea_endpoint() or os.getenv("ALDEA_API_URL") or "https://api.aldea.ai/asr/transcribe"
                token = os.getenv("ALDEA_API_KEY")
                headers = {"Content-Type": "wave"}
                if token:
                    headers["Authorization"] = f"Bearer {token}"

                # Check audio duration before sending
                if use_url:
                    audio_duration = sample["row"].get("audio_length_s")
                else:
                    audio_duration = (
                        len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]
                    )

                if audio_duration is not None and audio_duration < 0.160:
                    print(f"Skipping audio duration {audio_duration}s")
                    return "."

                if use_url:
                    # Download audio then POST as raw wave bytes
                    audio_url = sample["row"]["audio"][0]["src"]
                    resp = requests.get(audio_url, timeout=REQUEST_TIMEOUT)
                    resp.raise_for_status()
                    audio_data = BytesIO(resp.content)
                    response = requests.post(
                        api_url,
                        headers=headers,
                        data=audio_data,
                        timeout=REQUEST_TIMEOUT,
                    )
                else:
                    with open(audio_file_path, "rb") as f:
                        response = requests.post(
                            api_url,
                            headers=headers,
                            data=f,
                            timeout=REQUEST_TIMEOUT,
                        )

                response.raise_for_status()
                text = ""
                try:
                    payload = response.json()
                    if isinstance(payload, dict):
                        if isinstance(payload.get("text"), str):
                            text = payload["text"]
                        elif isinstance(payload.get("transcript"), str):
                            text = payload["transcript"]
                        elif isinstance(payload.get("data"), dict):
                            data_obj = payload["data"]
                            if isinstance(data_obj.get("text"), str):
                                text = data_obj["text"]
                except ValueError:
                    # Not JSON, fall back to raw text
                    text = response.text or ""

                return (text or "").strip()

            else:
                raise ValueError(
                    "Invalid model prefix, must start with 'assembly/', 'openai/', 'elevenlabs/', 'revai/', 'speechmatics/', 'deepgram/', 'groq/', or 'aldea/'"
                )

        except Exception as e:
            retries += 1
            if retries > max_retries:
                raise e

            if not use_url:
                sf.write(
                    audio_file_path,
                    sample["audio"]["array"],
                    sample["audio"]["sampling_rate"],
                    format="WAV",
                )
            
            # Use exponential backoff for rate limit errors
            if "429" in str(e) or "Too Many Requests" in str(e):
                delay = min(60, 2 ** retries)  # Cap at 60 seconds
                print(f"Rate limit error: {str(e)}. Retrying in {delay}s... (Attempt {retries}/{max_retries})")
            else:
                delay = 1
                print(f"API Error: {str(e)}. Retrying in {delay}s... (Attempt {retries}/{max_retries})")
            
            time.sleep(delay)


def transcribe_dataset(
    dataset_path,
    dataset,
    split,
    model_name,
    use_url=False,
    max_samples=None,
    max_workers=4,
):
    # Prepare results manifest path and resume state
    results_dir = "./results/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    manifest_path = os.path.join(
        results_dir,
        f"MODEL_{model_name.replace('/', '-')}_DATASET_{dataset_path.replace('/', '-')}_{dataset}_{split}.jsonl",
    )

    existing_count = 0
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as existing_file:
                existing_count = sum(1 for _ in existing_file)
        except Exception:
            existing_count = 0
        if existing_count > 0:
            print(
                f"Resuming existing results: found {existing_count} lines at {manifest_path}. Skipping those samples."
            )
    if use_url:
        audio_rows = fetch_audio_urls(dataset_path, dataset, split)
        if max_samples:
            audio_rows = itertools.islice(audio_rows, max_samples)
        ds = audio_rows
    else:
        ds = datasets.load_dataset(dataset_path, dataset, split=split, streaming=False)
        ds = data_utils.prepare_data(ds)
        if max_samples:
            ds = ds.take(max_samples)

    # Apply resume skipping if a manifest already exists
    if existing_count > 0:
        ds = itertools.islice(ds, existing_count, None)

    # Open manifest for append; write per-sample lines to avoid data loss on crash
    manifest_fp = open(manifest_path, "a", encoding="utf-8")
    write_index = existing_count
    file_write_lock = threading.Lock()

    print(f"Transcribing with model: {model_name}")

    def process_sample(sample):
        if use_url:
            reference = sample["row"]["text"].strip() or " "
            audio_duration = sample["row"]["audio_length_s"]
            start = time.time()
            try:
                transcription = transcribe_with_retry(
                    model_name, None, sample, use_url=True
                )
            except Exception as e:
                print(f"Failed to transcribe after retries: {e}")
                return None

        else:
            reference = sample.get("norm_text", "").strip() or " "
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                sf.write(
                    tmpfile.name,
                    sample["audio"]["array"],
                    sample["audio"]["sampling_rate"],
                    format="WAV",
                )
                tmp_path = tmpfile.name
                audio_duration = (
                    len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]
                )

            start = time.time()
            try:
                transcription = transcribe_with_retry(
                    model_name, tmp_path, sample, use_url=False
                )
            except Exception as e:
                print(f"Failed to transcribe after retries: {e}")
                os.unlink(tmp_path)
                return None
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                else:
                    print(f"File {tmp_path} does not exist")

        transcription_time = time.time() - start
        return reference, transcription, audio_duration, transcription_time

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sample = {executor.submit(process_sample, sample): sample for sample in ds}
        for future in tqdm(
            concurrent.futures.as_completed(future_to_sample),
            total=len(future_to_sample),
            desc="Transcribing",
        ):
            result = future.result()
            if not result:
                continue

            reference, transcription, audio_duration, transcription_time = result

            # None-safe coercion before normalization
            def to_text(value):
                if isinstance(value, str):
                    return value
                if value is None:
                    return " "
                return str(value)

            normalized_reference = data_utils.normalizer(to_text(reference)) or " "
            normalized_transcription = data_utils.normalizer(to_text(transcription)) or " "

            # Stream append one JSON line per completed sample
            with file_write_lock:
                manifest_line = {
                    "audio_filepath": f"sample_{write_index}",
                    "duration": audio_duration,
                    "time": transcription_time,
                    "text": normalized_reference,
                    "pred_text": normalized_transcription,
                }
                manifest_fp.write(f"{json.dumps(manifest_line, ensure_ascii=False)}\n")
                manifest_fp.flush()
                write_index += 1

    # Close the manifest writer before metrics
    try:
        manifest_fp.close()
    except Exception:
        pass

    print("Results saved at path:", manifest_path)

    # Compute metrics from the full manifest on disk (supports resume)
    try:
        manifest_data = eval_utils.read_manifest(manifest_path)
        references = [d.get("text", " ") for d in manifest_data]
        predictions = [d.get("pred_text", " ") for d in manifest_data]
        times = [d.get("time") for d in manifest_data]
        durations = [d.get("duration") for d in manifest_data]

        wer_metric = evaluate.load("wer")
        wer = wer_metric.compute(references=references, predictions=predictions)
        wer_percent = round(100 * wer, 2)

        compute_rtfx = (
            all(t is not None and t > 0 for t in times)
            and all(d is not None and d > 0 for d in durations)
        )
        if compute_rtfx:
            rtfx = round(sum(durations) / sum(times), 2)
            print("WER:", wer_percent, "%")
            print("RTFx:", rtfx)
        else:
            print("WER:", wer_percent, "%")
    except Exception as e:
        print(f"Could not compute metrics from manifest due to error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified Transcription Script with Concurrency"
    )
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--model_name",
        required=True,
        help="Prefix model name with 'assembly/', 'openai/', 'elevenlabs/', 'revai/', 'speechmatics/', 'deepgram/', or 'groq/'",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--max_workers", type=int, default=300, help="Number of concurrent threads"
    )
    parser.add_argument(
        "--use_url",
        action="store_true",
        help="Use URL-based audio fetching instead of datasets",
    )
    parser.add_argument(
        "--aldea_endpoints",
        type=str,
        default=None,
        help="Comma/space separated Aldea endpoints. Each may be a full URL or host:port."
    )

    args = parser.parse_args()

    # Initialize Aldea endpoints from CLI or environment
    aldea_arg = args.aldea_endpoints
    aldea_env = os.getenv("ALDEA_ENDPOINTS")
    endpoints_raw = None
    if aldea_arg and aldea_arg.strip():
        endpoints_raw = aldea_arg
    elif aldea_env and aldea_env.strip():
        endpoints_raw = aldea_env
    if endpoints_raw:
        # split by comma or whitespace
        parts = [p for chunk in endpoints_raw.split(",") for p in chunk.split()] if "," in endpoints_raw else endpoints_raw.split()
        set_aldea_endpoints(parts)

    transcribe_dataset(
        dataset_path=args.dataset_path,
        dataset=args.dataset,
        split=args.split,
        model_name=args.model_name,
        use_url=args.use_url,
        max_samples=args.max_samples,
        max_workers=args.max_workers,
    )
