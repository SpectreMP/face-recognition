# -*- coding: utf-8 -*-
import io
import os
import subprocess
import wave

import pytest

from voice_cache import VoiceCache, format_template


def make_wav_bytes(frames=b"\x00\x00" * 100):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(frames)
    return buf.getvalue()


@pytest.fixture
def fake_synth(monkeypatch):
    """Подменяет субпроцессный синтез. Сценарии: список из
    'ok' | 'empty' | 'raise' | 'timeout', по одному на попытку."""
    def install(script):
        actions = list(script)

        def fake(self, text, out_path):
            action = actions.pop(0) if actions else "ok"
            if action == "raise":
                raise RuntimeError("tts exploded")
            if action == "timeout":
                raise subprocess.TimeoutExpired(cmd="tts", timeout=30)
            if action == "empty":
                with open(out_path, "wb") as f:
                    f.write(b"")
                return
            with open(out_path, "wb") as f:
                f.write(make_wav_bytes())

        monkeypatch.setattr(VoiceCache, "_synthesize_in_subprocess", fake)

    return install


def test_format_template():
    assert format_template("Здравствуйте, {name}!", "Анна") == "Здравствуйте, Анна!"
    assert format_template("Привет, {name}. Как дела, {name}?", "Боб") == \
        "Привет, Боб. Как дела, Боб?"
    assert format_template("Сломанный {unknown_tag}", "X") == "Сломанный {unknown_tag}"


def test_synthesize_and_cache(fake_synth, tmp_path):
    fake_synth(["ok"])
    vc = VoiceCache(cache_dir=str(tmp_path), template="Здравствуйте, {name}!")
    assert vc.status("Анна") == "missing"
    assert vc.ensure("Анна", wait=True) == "ready"
    data = vc.get("Анна")
    assert data is not None
    assert data[:4] == b"RIFF"
    assert vc.status("Анна") == "ready"


def test_cached_file_survives_restart(fake_synth, tmp_path):
    fake_synth(["ok"])
    vc = VoiceCache(cache_dir=str(tmp_path))
    vc.ensure("Анна", wait=True)
    path = vc.wav_path("Анна")
    # "перезапуск": новый объект поверх того же каталога, без синтеза
    vc2 = VoiceCache(cache_dir=str(tmp_path))
    vc2._synthesize_in_subprocess = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("не должен синтезировать повторно"))
    assert os.path.exists(path)
    assert vc2.get("Анна") is not None
    assert vc2.status("Анна") == "ready"


def test_invalidate(fake_synth, tmp_path):
    fake_synth(["ok"])
    vc = VoiceCache(cache_dir=str(tmp_path))
    vc.ensure("Анна", wait=True)
    assert vc.status("Анна") == "ready"
    vc.invalidate("Анна")
    assert vc.get("Анна") is None
    assert vc.status("Анна") == "missing"


def test_retry_after_empty_file(fake_synth, tmp_path):
    fake_synth(["empty", "ok"])
    vc = VoiceCache(cache_dir=str(tmp_path))
    assert vc.ensure("Боб", wait=True) == "ready"
    assert vc.get("Боб") is not None


def test_retry_after_timeout(fake_synth, tmp_path):
    fake_synth(["timeout", "ok"])
    vc = VoiceCache(cache_dir=str(tmp_path), synth_timeout=5)
    assert vc.ensure("Боб", wait=True) == "ready"


def test_persistent_failure_reports_error(fake_synth, tmp_path):
    fake_synth(["raise", "timeout"])
    vc = VoiceCache(cache_dir=str(tmp_path))
    assert vc.ensure("Кэрол", wait=True) == "error"
    assert vc.status("Кэрол") == "error"


def test_template_change_changes_cache_key(fake_synth, tmp_path):
    fake_synth(["ok"])
    v1 = VoiceCache(cache_dir=str(tmp_path), template="Здравствуйте, {name}!")
    v2 = VoiceCache(cache_dir=str(v1.cache_dir), template="Приветствую, {name}!")
    assert v1.wav_path("Анна") != v2.wav_path("Анна")


def test_ensure_is_idempotent_when_ready(fake_synth, tmp_path):
    fake_synth(["ok"])
    vc = VoiceCache(cache_dir=str(tmp_path))
    vc.ensure("Дэйв", wait=True)

    def must_not_run(self, text, out_path):
        raise AssertionError("повторный синтез не нужен")

    import voice_cache as mod
    saved = mod.VoiceCache._synthesize_in_subprocess
    mod.VoiceCache._synthesize_in_subprocess = must_not_run
    try:
        assert vc.ensure("Дэйв") == "ready"
    finally:
        mod.VoiceCache._synthesize_in_subprocess = saved


def test_real_tts_subprocess_integration(tmp_path):
    """Реальный синтез (SAPI5/espeak-ng). Пропускается, если TTS недоступен."""
    pyttsx3 = pytest.importorskip("pyttsx3")
    del pyttsx3
    vc = VoiceCache(cache_dir=str(tmp_path), template="Здравствуйте, {name}!",
                    synth_timeout=60)
    status = vc.ensure("Михаил", wait=True, timeout=90)
    assert status == "ready"
    data = vc.get("Михаил")
    assert data is not None and len(data) > 1000
