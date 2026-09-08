# -*- coding: utf-8 -*-
import base64
import os

import client
from client import handle_command, play_wav_bytes, resolve_player


def test_resolve_player_none_available(monkeypatch):
    monkeypatch.setattr(client.shutil, "which", lambda name: None)
    assert resolve_player(None) is None


def test_resolve_player_first_available(monkeypatch):
    monkeypatch.setattr(client.shutil, "which",
                        lambda name: "/usr/bin/aplay" if name == "aplay" else None)
    assert resolve_player(None) == ["aplay", "-q"]


def test_resolve_player_explicit_cmd(monkeypatch):
    monkeypatch.setattr(client.shutil, "which",
                        lambda name: "/usr/bin/mpv" if name == "mpv" else None)
    assert resolve_player("mpv --really-quiet") == ["mpv", "--really-quiet"]


def test_resolve_player_explicit_missing_falls_back(monkeypatch):
    calls = []
    def fake_which(name):
        calls.append(name)
        return "/usr/bin/aplay" if name == "aplay" else None
    monkeypatch.setattr(client.shutil, "which", fake_which)
    result = resolve_player("nosuchplayer42")
    assert result == ["aplay", "-q"]
    assert "nosuchplayer42" in calls


def test_play_wav_bytes_runs_player(monkeypatch, wav_bytes):
    launched = {}
    class FakePopen:
        def __init__(self, cmd, **kwargs):
            launched["cmd"] = cmd

    tmpdir = os.path.join(os.getcwd(), "_tmp_client_test")
    os.makedirs(tmpdir, exist_ok=True)
    monkeypatch.setattr(client.tempfile, "mkstemp",
                        lambda suffix=None, prefix=None: (os.open(
                            os.path.join(tmpdir, "g.wav"), os.O_CREAT | os.O_WRONLY), 
                            os.path.join(tmpdir, "g.wav")))
    monkeypatch.setattr(client.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(client.shutil, "which", lambda name: "/usr/bin/paplay" if name == "paplay" else None)

    ok = play_wav_bytes(wav_bytes, player_cmd=None)
    assert ok is True
    player_cmd = launched["cmd"]
    assert player_cmd[0] == "paplay"
    tmp_file = player_cmd[-1]
    with open(tmp_file, "rb") as f:
        assert f.read() == wav_bytes
    os.remove(tmp_file)


def test_play_wav_bytes_no_player(monkeypatch, wav_bytes):
    monkeypatch.setattr(client.shutil, "which", lambda name: None)
    assert play_wav_bytes(wav_bytes) is False


def test_handle_command_greet_with_audio(monkeypatch, wav_bytes):
    received = {}

    def fake_play(data, player_cmd=None):
        received["data"] = data
        received["player"] = player_cmd
        return True

    monkeypatch.setattr(client, "play_wav_bytes", fake_play)
    msg = {
        "action": "greet",
        "name": "Михаил",
        "text": "Здравствуйте, Михаил!",
        "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
    }
    result = handle_command(msg, play=True, player_cmd=["aplay"])
    assert result["played"] is True
    assert received["data"] == wav_bytes


def test_handle_command_greet_without_audio():
    msg = {"action": "greet", "name": "Анна", "text": "Здравствуйте, Анна!",
           "audio_b64": None}
    result = handle_command(msg, play=True)
    assert result["played"] is False


def test_handle_command_audio_skipped_when_disabled(monkeypatch, wav_bytes):
    called = {"n": 0}

    def fake_play(data, player_cmd=None):
        called["n"] += 1
        return True

    monkeypatch.setattr(client, "play_wav_bytes", fake_play)
    msg = {"action": "greet", "name": "X", "audio_b64":
           base64.b64encode(wav_bytes).decode("ascii")}
    handle_command(msg, play=False)
    assert called["n"] == 0


def test_handle_command_unknown_action():
    result = handle_command({"action": "custom_cmd", "x": 1})
    assert result["action"] == "custom_cmd"
