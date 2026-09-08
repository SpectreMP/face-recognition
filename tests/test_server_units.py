# -*- coding: utf-8 -*-
import base64
import json

import numpy as np
import pytest
import zmq

from server import build_greet_command, recognize_face, send_command


def vec(first):
    v = np.zeros(128, dtype=np.float32)
    v[0] = first
    return v


ENTRIES = [("Анна", vec(0.1)), ("Михаил", vec(0.5))]


def test_recognize_match():
    name, dist = recognize_face(ENTRIES, vec(0.11), threshold=0.6)
    assert name == "Анна"
    assert dist < 0.6


def test_recognize_unknown():
    name, dist = recognize_face(ENTRIES, vec(0.48), threshold=0.6)
    assert name == "Михаил"  # ближе к Михаилу


def test_recognize_above_threshold_returns_none():
    name, dist = recognize_face(ENTRIES, vec(3.0), threshold=0.6)
    assert name is None
    assert dist > 0.6


def test_recognize_empty_entries():
    name, dist = recognize_face([], vec(0.0), threshold=0.6)
    assert name is None
    assert dist == 1.0


def test_build_greet_command_with_audio(wav_bytes):
    payload = build_greet_command("Михаил", wav_bytes, "Здравствуйте, {name}!")
    assert payload["action"] == "greet"
    assert payload["name"] == "Михаил"
    assert payload["text"] == "Здравствуйте, Михаил!"
    assert base64.b64decode(payload["audio_b64"]) == wav_bytes


def test_build_greet_command_without_audio():
    payload = build_greet_command("Анна", None, "Здравствуйте, {name}!")
    assert payload["audio_b64"] is None
    assert payload["text"] == "Здравствуйте, Анна!"


def test_send_command_zmq_roundtrip(wav_bytes):
    ctx = zmq.Context()
    try:
        pub = ctx.socket(zmq.PUB)
        port = pub.bind_to_random_port("tcp://127.0.0.1")
        sub = ctx.socket(zmq.SUB)
        sub.connect(f"tcp://127.0.0.1:{port}")
        sub.setsockopt_string(zmq.SUBSCRIBE, "clientX")
        import time
        time.sleep(0.3)

        payload = build_greet_command("Михаил", wav_bytes, "Здравствуйте, {name}!")
        send_command(pub, "clientX", payload)

        if not sub.poll(3000):
            pytest.fail("Команда не доставлена по PUB/SUB")
        topic = sub.recv_string()
        body = json.loads(sub.recv_string())
        assert topic == "clientX"
        assert body["action"] == "greet"
        assert body["name"] == "Михаил"
        assert base64.b64decode(body["audio_b64"]) == wav_bytes
    finally:
        pub.close(0)
        sub.close(0)
        ctx.term()
