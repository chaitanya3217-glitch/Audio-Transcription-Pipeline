# Transcription Pipeline

A small, production-minded speech-to-text pipeline. It accepts audio in any common
format, transcribes it with an open-source model (**faster-whisper**), and returns
clean text with **per-segment timestamps** as JSON, SRT, VTT or plain text. It runs
as a **CLI** or as an **async REST API**.

The focus is on the engineering around the model: format handling, long audio,
failure recovery, and a clear path to scale.

```
            ┌────────┐   ┌───────────┐   ┌──────────────┐   ┌──────────────────┐   ┌────────────┐   ┌────────────┐
 any file ─►│ probe  │──►│ normalize │──►│ plan chunks  │──►│ transcribe chunk │──►│   merge    │──►│ postprocess│──► JSON / SRT / VTT / TXT
            │ffprobe │   │ ffmpeg →  │   │ cut at quiet │   │ retry + backoff  │   │ shift to   │   │ clean, drop│
            │validate│   │16k mono   │   │ points       │   │ checkpoint each  │   │ global time│   │ hallucin.  │
            └────────┘   │ PCM WAV   │   └──────────────┘   └──────────────────┘   └────────────┘   └────────────┘
                         └───────────┘
```

---

## Quick start

Requirements: Python 3.10+ and `ffmpeg`/`ffprobe` on `PATH`.

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements-dev.txt

# CLI
python -m transcriber transcribe samples/short.mp3                 # JSON to stdout
python -m transcriber transcribe samples/short.wav -f srt -o out.srt
python -m transcriber transcribe samples/long.mp3 --chunk-seconds 60 -f txt
python -m transcriber transcribe talk.m4a --model small --language en --word-timestamps

# API
python -m transcriber serve --port 8000
curl -F file=@samples/short.mp3 http://localhost:8000/v1/transcriptions          # -> 202 {id, status_url}
curl http://localhost:8000/v1/transcriptions/<id>                                 # status / progress / result
curl "http://localhost:8000/v1/transcriptions/<id>/transcript?format=srt"         # download as subtitles
# interactive docs: http://localhost:8000/docs

# Tests (24 tests, ~10s; use a mock engine so no model download needed)
pytest -q

# Docker
docker build -t transcriber . && docker run -p 8000:8000 -v $PWD/data:/data transcriber
```

The first real run downloads the Whisper model (~140 MB for `base`) from Hugging Face.

### Example output

`samples/` has test audio generated with a text-to-speech engine (mock data).
`examples/` has the actual output of the pipeline on that audio.

```json
{
  "segments": [
    { "id": 0, "start": 0.0,  "end": 2.96,  "text": "Hello and welcome to the transcription pipeline demo.", ... },
    { "id": 1, "start": 3.76, "end": 7.04,  "text": "This audio file was generated with a text to speech engine.", ... },
    { "id": 2, "start": 7.84, "end": 13.84, "text": "The pipeline converts audio into text, and returns time stamps for every segment.", ... }
  ],
  "language": "en",
  "duration": 13.952,
  "source":   { "format_name": "mp3", "codec": "mp3", "sample_rate": 22050, "channels": 1, ... },
  "metadata": { "engine": "faster-whisper:base", "chunks": 1, "processing_seconds": 2.52, "real_time_factor": 0.18 },
  "text": "Hello and welcome to the transcription pipeline demo. This audio file was ..."
}
```

---

## Code layout

| File | Responsibility |
|---|---|
| `transcriber/audio.py` | `probe` (ffprobe validation), `normalize` (ffmpeg → 16 kHz mono WAV), `read_window` (seek-and-read a time range), `plan_chunks` (silence-aware splitting) |
| `transcriber/engine.py` | `Engine` interface, `WhisperEngine` (faster-whisper), `MockEngine` (tests) |
| `transcriber/pipeline.py` | Orchestration: chunk loop, retries with backoff, per-chunk checkpoints, timestamp shifting, language pinning |
| `transcriber/postprocess.py` | Text cleanup, hallucination/repetition filtering, JSON/SRT/VTT/TXT formatters |
| `transcriber/jobs.py` | Durable job store + bounded worker pool + crash recovery |
| `transcriber/api.py` | FastAPI app (upload, status, download) |
| `transcriber/cli.py` | `transcribe` and `serve` commands |
| `transcriber/errors.py` | `PermanentError` vs `TransientError`, which drives retry behaviour |

---

# Part 1: Transcription pipeline

### Accepting an audio file
- **CLI:** `python -m transcriber transcribe <file>`.
- **API:** `POST /v1/transcriptions` (multipart upload). The upload is **streamed to disk in
  1 MB pieces**, so a 500 MB file never sits in RAM. There is a configurable size cap (413) and
  the file is **validated immediately with ffprobe**, so garbage is rejected with a 415 straight
  away instead of failing minutes later in a worker.

### Transcribing speech to text
- **Model: [faster-whisper](https://github.com/SYSTRAN/faster-whisper)**, a CTranslate2 port of
  OpenAI Whisper. I chose it because it is:
  - open source and runs **offline**, so no audio leaves the machine (privacy, no per-minute cost);
  - multilingual with automatic language detection;
  - ~4× faster than reference Whisper, and **int8 quantization** makes CPU-only use practical.
    On my laptop (no GPU), `base` runs at about **0.18× real time**: 4 minutes of audio in about 40 s.
- **Model size is a config knob** (`tiny` → `large-v3`), so you can trade accuracy for speed per
  deployment.

  Measured on my laptop CPU (no GPU, int8), on the 3.6-minute `samples/long.mp3`:

  | Model | Download | Time | Real-time factor | Result |
  |---|---|---|---|---|
  | `tiny` | 75 MB | 31 s | 0.14× | Text mostly right; minor slips ("time line", a dropped "The") and less consistent segment boundaries |
  | `base` (default) | 140 MB | 40 s | 0.18× | All 60 sentences correct |
  | `small` | 460 MB | 171 s | 0.78× | All 60 sentences correct |

  `base` is the default: on a CPU it's barely slower than `tiny` and noticeably more reliable.
  `small` is worth it for noisy real-world audio. `medium` and `large-v3` really need a GPU.
- **Decoding settings, and why:**
  - `vad_filter=True`: Silero VAD skips non-speech. It is faster and is the single best defence
    against Whisper "hallucinating" text on silence or music.
  - `condition_on_previous_text=False`: stops one bad segment from triggering a repetition loop
    in all the segments after it.
  - `beam_size=5`: the standard accuracy/speed balance.
- **Engine abstraction.** The pipeline only knows `engine.transcribe(audio, language)`. Moving to a
  GPU server or a hosted API (Deepgram, AssemblyAI, OpenAI, Google) means writing one class.
  Tests use a `MockEngine` through the same interface.

### Returning timestamps per segment
- Every segment has `start`/`end` in seconds **on the original file's timeline**. Chunk-relative
  timestamps are shifted by the chunk's offset during merging.
- `--word-timestamps` (or `word_timestamps=True`) also gives word-level timing, which is useful
  for search highlighting or karaoke-style captions.
- **Post-processing for downstream use** (`postprocess.py`):
  - normalizes whitespace and punctuation spacing;
  - drops empty segments;
  - drops likely hallucinations (Whisper's own heuristic: `no_speech_prob > 0.6` **and** `avg_logprob < -1.0`);
  - collapses consecutive duplicate segments (repetition loops);
  - enforces monotonic, non-overlapping timestamps and renumbers ids.

  Confidence fields (`avg_logprob`, `no_speech_prob`) are kept, so a downstream consumer can flag
  low-confidence spans for human review.
- **Output formats:** JSON (machine use), SRT and VTT (subtitles/players), TXT (humans).

### How do you handle different audio formats?
**Normalize once, at the edge, into one canonical format.**

1. **Don't trust the extension.** `ffprobe` inspects the actual content. A file with no audio
   stream, a corrupt file, or an empty file fails fast with `UnsupportedAudioError`, which is a
   *permanent* error, so it is never retried. A `.wav` that is really an MP3 still works.
2. **Decode everything with ffmpeg** into **16 kHz, mono, 16-bit PCM WAV**, which is exactly what
   Whisper expects. This covers WAV, MP3, M4A/AAC, FLAC, OGG/Opus, WebM, WMA, AMR, AIFF, and the
   audio track of video files (MP4/MKV/MOV, using `-vn`). Stereo is down-mixed and any sample rate
   is resampled.
3. Everything downstream (chunking, the engine, retries) only ever sees that one format. Format
   bugs can only live in one function, and the WAV is **randomly seekable**, which is what makes
   cheap chunking possible (see below).
4. The normalized file is written to a temp name and **atomically renamed**, so a crash can't
   leave a half-written file that a resumed job would trust.

The original upload is kept untouched, so re-processing with a better model later is always possible.
Tests cover MP3, FLAC, OGG/Vorbis, M4A/AAC, 44.1 kHz stereo WAV, a mislabelled file, and non-audio input.

### How do you deal with long audio files?
Whisper works on 30-second windows internally, but feeding it a 3-hour file in one call means the
whole decoded file sits in memory, there's no progress, and one crash at 2h59m loses everything.
So the pipeline **splits long audio into chunks** (default 10 minutes):

- **Silence-aware cuts.** For each boundary, the code looks back over the previous 30 s, computes
  frame RMS energy, smooths it, and **cuts at the quietest point**, normally a pause between
  sentences. Cutting in silence means words aren't split, so chunks **don't need to overlap** and
  there's no fragile text de-duplication step. (`audio.plan_chunks`, `audio.quietest_offset`.)
  - *Alternative considered:* fixed windows with overlap, de-duplicated by "a segment belongs to the
    chunk that contains its midpoint". It works, but segments that straddle the overlap can come out
    differently in each chunk, so you get partial duplicates or drops. Silence cuts avoid the problem
    instead of patching it.
  - *Verified:* a 3.6-minute file of 12 repeated paragraphs, cut into 6 chunks, comes back with every
    sentence exactly 12 times (no loss or duplication at 5 boundaries). See `examples/long_chunked.txt`.
- **Bounded memory.** `read_window` seeks into the WAV and loads only the current chunk, so memory
  is constant whether the file is 1 minute or 10 hours.
- **Checkpoint per chunk.** Each finished chunk is written atomically to `work/chunks/NNNN.json`.
  Re-running a job **resumes at the first missing chunk**: deleting 1 of 7 checkpoints and re-running
  took 7 s instead of 42 s. A checkpoint is only reused if its chunk boundaries match, so changing
  `chunk_seconds` can't mix results.
- **Retry at chunk granularity.** A transient failure re-does 10 minutes of audio, not 3 hours.
- **Language pinning.** The language is detected on the first chunk, then passed explicitly to the
  later ones. Output is consistent and language ID isn't re-run every time.
- **Progress.** Each finished chunk updates `progress` on the job, which the API exposes.
- **Parallelism hook.** Chunks are independent, so in the distributed design below they become
  separate tasks that can run on different GPUs at the same time.

---

# Part 2: System design (not implemented; how I'd take this to production)

The local implementation already has the same *shape* as the production design: an upload goes to
storage, a job record is created, a worker processes it with checkpoints, and the client polls for
the result. Scaling it out is mostly swapping backends, not rewriting logic.

```
                ┌──────────────┐  presigned PUT   ┌───────────────────┐
  Client ──────►│  API service │◄────────────────►│ Object storage    │
    ▲  │        │ (stateless,  │                  │ (S3/GCS)          │
    │  │        │  autoscaled) │                  │ raw/ normalized/  │
    │  │        └──────┬───────┘                  │ chunks/ results/  │
    │  │               │ insert job               └─────────▲─────────┘
    │  │               ▼                                    │
    │  │        ┌──────────────┐   enqueue    ┌─────────────┴──────────┐
    │  │        │  Postgres    │─────────────►│  Queue (SQS / Pub/Sub) │──► DLQ
    │  │        │ jobs, chunks,│              └─────────────┬──────────┘
    │  │        │ segments     │◄───status──┐               │ pull
    │  │        └──────────────┘            │   ┌───────────▼───────────┐
    │  └──── webhook / SSE ◄────────────────┴───│ Workers               │
    └──── GET /v1/transcriptions/{id}           │  prep (CPU): probe,   │
                                                │   normalize, split    │
                                                │  asr (GPU): per chunk │
                                                │  merge (CPU)          │
                                                └───────────────────────┘
```

### How would you handle concurrent uploads?
- **Decouple ingest from processing.** The API only accepts the upload, creates a job and returns
  `202 Accepted`. Transcription happens on workers pulling from a queue. Upload spikes become queue
  depth, not timeouts or OOMs.
- **Uploads go straight to object storage** using **presigned URLs** (multipart for large files),
  so API servers never proxy bytes. The client calls `POST /uploads` to get a URL, PUTs the file,
  then calls `POST /transcriptions {upload_id}`. API pods stay small and stateless.
- **Bounded concurrency + backpressure.** Each GPU worker runs a fixed number of model instances
  (here: `TRANSCRIBER_WORKERS` threads and CTranslate2 `num_workers`). When the queue backs up,
  the API returns `429 Retry-After` (implemented here with `MAX_QUEUE`) and the autoscaler adds
  workers, scaling on **queue depth / oldest message age**, not CPU.
- **Split CPU and GPU work.** ffmpeg decoding and splitting run on cheap CPU workers. Only chunk
  inference uses GPUs. Chunks fan out across GPUs, so one 5-hour file doesn't block 100 short ones.
  **Batching** many short chunks per GPU call (faster-whisper's `BatchedInferencePipeline`) raises
  throughput.
- **Fairness.** Per-tenant rate limits and quotas, plus separate queues or priority lanes
  (e.g. interactive short clips vs. bulk backfills), so one customer can't starve others.
- **Idempotency.** Clients send an `Idempotency-Key`, and the content hash (SHA-256) of the upload
  is recorded. A double-submit returns the existing job, and identical audio can reuse a cached
  transcript.

*Implemented locally:* streaming uploads, async jobs, a bounded `ThreadPoolExecutor`, 429 on a full
queue, and a test that submits 5 concurrent jobs.

### How would you store audio and transcripts?
- **Audio → object storage** (S3/GCS/Azure Blob). It's cheap, durable and big-file friendly.
  - `raw/{job_id}/input.ext`: the original, immutable, so it can be re-processed with a better model later;
  - `work/{job_id}/audio.wav` and `chunks/`: intermediates, deleted by a **lifecycle rule** after N days;
  - raw audio moves to infrequent-access/archive storage after 30 days, then is deleted according to
    the **retention policy** (audio is often PII: think voice biometrics, medical, call-centre).
- **Metadata and transcripts → Postgres.**
  - `jobs(id, tenant_id, status, attempts, progress, error, model, language, input_uri, sha256, created_at, ...)`;
  - `chunks(job_id, idx, start, end, status, attempts, result_uri)` drives fan-out and resume;
  - `segments(job_id, idx, start, end, text, confidence, speaker)` for queryable, paginated access;
  - the full transcript JSON is also stored in object storage (`results/{job_id}.json`) for cheap bulk reads.
- **Search.** Segments are indexed into OpenSearch/Elasticsearch (full-text plus timestamp jump-to),
  and optionally embedded into a vector store for semantic search/RAG.
- **Version transcripts by model.** Store `model` and `model_version` on each transcript, so
  re-running with `large-v3` adds a new version instead of overwriting.
- **Security.** Encryption at rest (KMS) and in transit, per-tenant prefixes / row-level security,
  short-lived presigned download URLs, and audit logs.

*Implemented locally:* the same layout on the filesystem (`data/jobs/<id>/{input, work/, transcript.json, job.json}`),
with every write atomic (temp file + rename).

### How do you retry or recover failed transcriptions?
Layered, from cheapest to most drastic:

1. **Classify errors first.** `PermanentError` (corrupt/unsupported audio, no audio stream) fails the
   job immediately with a clear message, because retrying wastes compute. `TransientError` (GPU OOM,
   worker crash, network/storage blip, API 5xx/429) is retried.
2. **Chunk-level retry** with exponential backoff (+ jitter in production): 3 attempts by default.
   Only the failing 10-minute chunk is redone.
3. **Checkpoints make retries and crashes cheap.** Finished chunks are persisted, so any retry or
   restart resumes from the first missing chunk. Writes are atomic, so a crash can't leave a corrupt
   checkpoint, and an unreadable one is simply redone.
4. **Job-level retry** (`MAX_JOB_ATTEMPTS`) for failures outside the chunk loop. Thanks to (3), it
   doesn't redo finished work.
5. **Crash recovery.** On startup the service re-enqueues any job left `queued` or `processing`, and
   it resumes from its checkpoints. In production the queue does this for you: a message isn't acked
   until the job finishes, and a worker that dies stops extending its **visibility timeout**
   (heartbeat), so the message reappears for another worker. Workers are **idempotent** (the same
   job id writes the same keys), so a duplicate delivery is harmless.
6. **Dead-letter queue.** After N deliveries, the message goes to a DLQ with the error attached. Then
   there's an alert, a dashboard and a one-click "redrive" after a fix. Jobs are marked `failed` with
   a human-readable `error`, and the client gets a webhook.
7. **Degrade instead of fail.** For example, retry a chunk that OOMs on `large-v3` with a smaller
   model or on CPU, and flag it as lower quality.
8. **Observability.** Metrics for queue depth, job latency, real-time factor, failure rate by error
   class, and retries. Traces carry the job id through API → queue → worker, so you can see *why*
   something failed, not just that it did.

*Implemented locally:* steps 1–5 with tests: `test_transient_failures_are_retried`,
`test_permanent_errors_are_not_retried`, `test_resume_reuses_finished_chunks`,
`test_recovers_unfinished_jobs_on_restart`.

### How would you expose this as an API?
An **asynchronous, job-based REST API** (implemented in `api.py`, OpenAPI docs at `/docs`):

| Method & path | Purpose |
|---|---|
| `POST /v1/transcriptions` | Multipart upload (or `{"audio_url"}` / `{"upload_id"}` in prod) + options (`language`, `model`, `word_timestamps`, `webhook_url`). Returns **202** with `id` and `status_url`. |
| `GET /v1/transcriptions/{id}` | `status` (`queued`/`processing`/`completed`/`failed`), `progress` (0–1), `attempts`, `error`, and the `result` when done. |
| `GET /v1/transcriptions/{id}/transcript?format=json\|srt\|vtt\|txt` | Download in the format the consumer needs. **409** if not finished. |
| `DELETE /v1/transcriptions/{id}` | *(prod)* Cancel, or delete audio + transcript (right to erasure). |
| `GET /healthz` | Liveness/readiness, plus the active engine. |

Design choices:
- **Async, not a blocking request.** A long file takes minutes, longer than load balancers and
  clients will hold a connection. Clients **poll**, or better, register a **webhook** (signed with
  HMAC, retried) for completion. For live progress, use **Server-Sent Events**.
- **Clear status codes.** 202 accepted, 413 too large, 415 not audio, 429 busy (+`Retry-After`),
  404 unknown job, 409 not ready yet.
- **Versioned path** (`/v1`), with an **API key / OAuth2** per tenant, rate limits at the gateway,
  and `Idempotency-Key` support.
- **Real-time use case** (live captions) is a different product: a **WebSocket/gRPC streaming**
  endpoint that takes audio frames and returns partial and final segments. It shares the engine
  abstraction but not the batch job machinery.
- **Pagination** of `segments` for very long transcripts (`?offset=&limit=`) so a 5-hour result
  isn't one 20 MB response.

---

## Trade-offs and what I'd do next
- **Single-node job store.** `jobs.py` uses the local disk plus a thread pool, which is good for one
  box and has the same interfaces as the distributed design. Next step: Postgres + SQS + S3 behind
  the same `JobManager` methods.
- **Chunks run sequentially within a job.** On CPU, CTranslate2 already uses all cores per chunk, so
  parallel chunks would only compete. On a GPU fleet I'd fan chunks out across workers.
- **Speaker diarization** (who spoke when) would be a post-processing step, e.g. pyannote. It aligns
  speaker turns to segment timestamps.
- **Evaluation.** Add a small labelled set and track **WER** in CI, so model or config changes are
  measured, not guessed.
- **Domain vocabulary.** Whisper's `initial_prompt` / hotwords can bias recognition toward product
  names and jargon.
