# -*- coding: utf-8 -*-
import io
import wave

import numpy as np
import pytest

import face_store


class FakeFaceRecognition:
    """Подменяет face_recognition для быстрых тестов без dlib."""

    def __init__(self):
        self.faces_present = True
        self.counter = 0

    def load_image_file(self, path):
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def face_locations(self, img, model="hog"):
        if not self.faces_present:
            return []
        return [(0, 4, 4, 0)]

    def face_encodings(self, img, boxes=None):
        self.counter += 1
        vec = np.zeros(128, dtype=np.float32)
        vec[0] = 0.001 * self.counter
        return [vec]


@pytest.fixture
def fake_fr(monkeypatch):
    fr = FakeFaceRecognition()
    monkeypatch.setattr(face_store, "face_recognition", fr)
    return fr


class DummyVoice:
    """Заглушка VoiceCache с тем же интерфейсом: сразу пишет готовый WAV."""

    def __init__(self, template="Здравствуйте, {name}!", fail=False):
        import tempfile
        self.template = template
        self.fail = fail
        self.calls = []
        self._tmpdir = tempfile.mkdtemp(prefix="dummy_voice_")

    def cleanup(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def text_for(self, name):
        return self.template.replace("{name}", name)

    def wav_path(self, name):
        import os
        import hashlib
        digest = hashlib.sha1(f"{self.template}|{name}".encode("utf-8")).hexdigest()[:16]
        return os.path.join(self._tmpdir, f"{digest}.wav")

    def get(self, name):
        try:
            with open(self.wav_path(name), "rb") as f:
                data = f.read()
            return data if len(data) > 44 else None
        except OSError:
            return None

    def status(self, name):
        if self.fail:
            return "error"
        return "ready" if self.get(name) is not None else "missing"

    def ensure(self, name, force=False, wait=False, timeout=30.0):
        self.calls.append(name)
        if self.fail:
            return "error"
        if force or self.get(name) is None:
            with open(self.wav_path(name), "wb") as f:
                f.write(make_wav_bytes(b"\x00\x00" * 100))
        return "ready"

    def invalidate(self, name):
        import os
        try:
            os.remove(self.wav_path(name))
        except OSError:
            pass


@pytest.fixture
def dummy_voice():
    v = DummyVoice()
    yield v
    v.cleanup()


@pytest.fixture
def failing_dummy_voice():
    v = DummyVoice(fail=True)
    yield v
    v.cleanup()


@pytest.fixture
def client_pair_factory(fake_fr, tmp_path):
    from face_store import FaceStore
    from webapp import create_app

    made_voices = []

    def make(voice=None):
        faces_dir = str(tmp_path / "faces")
        store = FaceStore(faces_dir)
        v = voice or DummyVoice()
        made_voices.append(v)
        app = create_app(store, v, faces_dir)
        return app.test_client(), store

    yield make
    for v in made_voices:
        if isinstance(v, DummyVoice):
            v.cleanup()


@pytest.fixture
def client_pair(client_pair_factory):
    return client_pair_factory()


def make_wav_bytes(pcm_frames=b"\x00\x00" * 2205, rate=22050):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm_frames)
    return buf.getvalue()


@pytest.fixture
def wav_bytes():
    return make_wav_bytes()
