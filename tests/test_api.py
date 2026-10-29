import time

import pytest
from fastapi.testclient import TestClient

from transcriber.api import create_app
from transcriber.config import Settings
from transcriber.engine import MockEngine


@pytest.fixture
def client(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", workers=2, chunk_seconds=30)
    with TestClient(create_app(settings, MockEngine())) as c:
        yield c


def wait_for(client, job_id, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/v1/transcriptions/{job_id}").json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.1)
    raise TimeoutError(body)


def test_upload_poll_and_download(client, speechy_wav, to_format):
    mp3 = to_format(speechy_wav, ".mp3")
    r = client.post("/v1/transcriptions", files={"file": ("talk.mp3", mp3.read_bytes())},
                    data={"language": "en"})
    assert r.status_code == 202
    job_id = r.json()["id"]

    body = wait_for(client, job_id)
    assert body["status"] == "completed" and body["progress"] == 1.0
    segs = body["result"]["segments"]
    assert segs and all({"start", "end", "text"} <= s.keys() for s in segs)

    srt = client.get(f"/v1/transcriptions/{job_id}/transcript?format=srt")
    assert srt.status_code == 200 and "-->" in srt.text


def test_concurrent_uploads(client, speechy_wav):
    data = speechy_wav.read_bytes()
    ids = [client.post("/v1/transcriptions", files={"file": (f"{i}.wav", data)}).json()["id"]
           for i in range(5)]
    assert len(set(ids)) == 5
    assert all(wait_for(client, i)["status"] == "completed" for i in ids)


def test_rejects_non_audio_with_415(client):
    r = client.post("/v1/transcriptions", files={"file": ("x.mp3", b"not audio" * 100)})
    assert r.status_code == 415


def test_unknown_job_404(client):
    assert client.get("/v1/transcriptions/does-not-exist").status_code == 404
    assert client.get("/v1/transcriptions/00000000-0000-0000-0000-000000000000").status_code == 404


def test_recovers_unfinished_jobs_on_restart(tmp_path, speechy_wav):
    """A job left 'processing' by a crashed server is picked up on the next boot."""
    settings = Settings(data_dir=tmp_path / "data", workers=1, chunk_seconds=30)
    app = create_app(settings, MockEngine())
    manager = app.state.manager
    from transcriber.jobs import Job, Status
    job_id, d = manager.new_job_dir()
    (d / "input.wav").write_bytes(speechy_wav.read_bytes())
    job = Job(id=job_id, filename="x.wav", input_path=str(d / "input.wav"),
              status=Status.PROCESSING, attempts=1)
    manager._save(job)                     # as if we crashed mid-run

    with TestClient(create_app(settings, MockEngine())) as client:  # "restart"
        assert wait_for(client, job_id)["status"] == "completed"
