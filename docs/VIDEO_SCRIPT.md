# Video walkthrough script (~7–8 minutes)

Screen-record your editor plus a terminal. **Bold** = what to show on screen. Speak naturally;
the lines are a guide, not something to read word for word.

---

### 1. Intro: the problem and my approach (45 s)
**Show: README.md, the pipeline diagram at the top.**

> "The task was to build a transcription pipeline: take an audio file, turn it into text with
> timestamps per segment, and make it usable downstream. The brief said the focus is on engineering
> decisions, not training a model. So I picked a solid open-source model and spent my effort on
> everything around it: handling any input format, long files, failures, and a clean path to an API.
>
> The pipeline has six stages: probe, normalize, plan chunks, transcribe each chunk with retries and
> checkpoints, merge onto one timeline, and post-process."

### 2. Live demo first (1 min)
**Terminal:**
```bash
python -m transcriber transcribe samples/short.mp3 -f txt
python -m transcriber transcribe samples/short.wav -f srt
python -m transcriber transcribe samples/long.mp3 --chunk-seconds 60 -f txt
```
> "Here's an MP3 and a WAV. Same output, with timestamps per segment. Now a 3.6-minute file. I'm
> forcing 60-second chunks so you can see the chunking: six chunks, and the transcript comes back on
> one continuous timeline. I checked: every sentence appears exactly the expected number of times,
> so nothing is lost or duplicated at the cuts."

### 3. Model choice (45 s)
**Show: `transcriber/engine.py`, the `WhisperEngine` class.**

> "For the model I used faster-whisper. It's Whisper re-implemented on CTranslate2: open source,
> runs offline so audio never leaves the machine, multilingual, and with int8 quantization it's fast
> enough on a CPU. My laptop has no GPU and it runs at about 0.18 times real time.
>
> Three decoding settings matter: the VAD filter skips silence, which is the main defence against
> Whisper hallucinating text. Not conditioning on previous text stops repetition loops. And beam
> size 5 is the usual accuracy/speed balance.
>
> Most importantly, the rest of the code only knows this tiny `Engine` interface. Switching to a GPU
> server or a hosted API is one new class. The tests use a `MockEngine` through the same interface."

### 4. Different formats (1 min)
**Show: `transcriber/audio.py`, `probe` and `normalize`.**

> "For formats, my rule was: normalize once, at the edge. First, ffprobe inspects the actual
> content. I don't trust the file extension. If there's no audio stream or the file is corrupt, it
> raises an UnsupportedAudioError. That's a *permanent* error, so we never waste retries on it, and
> the API rejects it with a 415 right at upload time.
>
> Then ffmpeg converts whatever came in, whether MP3, M4A, FLAC, OGG or even the audio track of an
> MP4, into one canonical format: 16 kHz mono 16-bit WAV, which is what Whisper expects. Everything
> downstream only ever sees that one format. It's written to a temp file and renamed atomically, so
> a crash can't leave a half-written file behind."

**Optionally show: `tests/test_audio.py`, the parametrized format test.**

### 5. Long audio (1 min 30 s)
**Show: `audio.py`, `quietest_offset` and `plan_chunks`.**

> "Long files were the most interesting decision. If you send a 3-hour file in one call, it all sits
> in memory, you get no progress, and a crash near the end loses everything. So I split into chunks,
> 10 minutes by default.
>
> The naive way is fixed windows with overlap and then de-duplicating text. But a sentence crossing
> the boundary can be transcribed slightly differently in each window, so you get duplicates or
> gaps. Instead, near each boundary I look back 30 seconds, compute the audio energy, and cut at the
> quietest point, usually a pause between sentences. Since the cut is in silence, the chunks don't
> need to overlap at all.
>
> `read_window` seeks into the WAV and loads only that chunk, so memory is constant regardless of
> file length."

**Show: `pipeline.py`, the `run` loop.**

> "In the pipeline loop, each chunk gets transcribed with retries, then its timestamps are shifted
> by the chunk's start time so they're relative to the original file, and the result is checkpointed
> to disk. The language detected on the first chunk is pinned for the rest, for consistency."

### 6. Reliability: retries and recovery (1 min)
**Show: `errors.py`, then `with_retries` in `pipeline.py`.**

> "Failures are split into permanent and transient. Permanent ones fail immediately. Transient ones,
> like an out-of-memory error, get retried with exponential backoff, at chunk level, so we only redo
> 10 minutes, not the whole file.
>
> And because every finished chunk is checkpointed, a crash or a retry resumes from the first missing
> chunk."

**Terminal demo (optional):**
```bash
python -m transcriber transcribe samples/long.mp3 --chunk-seconds 60 --work-dir data/demo -o out.json
del data\demo\chunks\0002.json          # simulate a crash
python -m transcriber -v transcribe samples/long.mp3 --chunk-seconds 60 --work-dir data/demo -o out.json
```
> "If I delete one checkpoint and re-run, only that chunk is processed. The metadata shows 5 chunks
> resumed."

**Show: `postprocess.py`.**
> "Post-processing cleans up whitespace, drops likely hallucinations using Whisper's own confidence
> signals, collapses repetition loops, and outputs JSON, SRT, VTT or plain text."

### 7. The API (1 min)
**Show: `api.py`, then the terminal.**
```bash
python -m transcriber serve --port 8000
curl -F file=@samples/short.mp3 localhost:8000/v1/transcriptions
curl localhost:8000/v1/transcriptions/<id>
curl "localhost:8000/v1/transcriptions/<id>/transcript?format=vtt"
```
**Also show: http://localhost:8000/docs in a browser.**

> "The API is asynchronous on purpose. Long files take minutes, which is longer than load balancers
> will hold a request. Upload returns 202 with a job id. The upload is streamed to disk in 1 MB
> pieces and validated straight away. You poll for status and progress, and download the transcript
> in whichever format you need.
>
> Behind it, `jobs.py` is a small durable job queue: job state on disk, a bounded worker pool, a 429
> when the queue is full, and on startup it re-enqueues any job that was interrupted. Those jobs
> resume from their checkpoints."

### 8. System design, Part 2 (1 min)
**Show: README, the Part 2 diagram.**

> "For production, the local design already has the right shape, so scaling is mostly swapping
> backends. Uploads go directly to S3 with presigned URLs so API servers stay stateless. Job and
> chunk state live in Postgres. A queue like SQS feeds workers. CPU workers handle ffmpeg, GPU
> workers handle inference, and chunks fan out across GPUs. Autoscaling is based on queue depth.
>
> For recovery, the queue only acks a message when the work is done. A dead worker's message
> reappears for another worker, the checkpoints mean nothing is redone, and after N attempts it goes
> to a dead-letter queue with an alert. Clients can poll or get a signed webhook.
>
> Audio is stored in object storage with lifecycle rules, since it's often PII. Transcripts are
> stored in Postgres and indexed for search, versioned by model."

### 9. Tests and wrap-up (30 s)
**Terminal:** `pytest -q` (24 passed).

> "There are 24 tests, covering formats, chunk planning, retries, resume, crash recovery, concurrent
> uploads and the API. They use the mock engine, so they run in about 10 seconds. Next steps would be
> speaker diarization, a WER evaluation set in CI, and swapping the local job store for the
> distributed one. Thanks for watching!"
