# -*- coding: utf-8 -*-
import hashlib
import logging
import os
import queue
import subprocess
import sys
import threading
import time

log = logging.getLogger(__name__)


def format_template(template, name):
    try:
        return template.format(name=name)
    except (KeyError, IndexError, ValueError):
        return template.replace("{name}", name)

# Синтез выполняется в отдельном процессе: зависший/упавший TTS не влияет на
# сервер, а таймаут гарантирует завершение. Работает и с SAPI5 (Windows),
# и с espeak-ng/speech-dispatcher (Linux).
TTS_SUBPROCESS_CODE = r"""
import sys
import pyttsx3

text, out_path = sys.argv[1], sys.argv[2]
rate = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None
hint = sys.argv[4] if len(sys.argv) > 4 else None

engine = pyttsx3.init()
if rate:
    engine.setProperty("rate", rate)
if hint:
    h = hint.lower()
    for v in engine.getProperty("voices"):
        ident = str(getattr(v, "id", ""))
        name = str(getattr(v, "name", ""))
        if h in ident.lower() or h in name.lower():
            engine.setProperty("voice", v.id)
            break
engine.save_to_file(text, out_path)
engine.runAndWait()
"""


class VoiceCache:
    """Кэш заранее синтезированных приветствий.

    Фразы синтезируются ЗАРАНЕЕ (при добавлении фото через веб-панель либо при
    старте сервера) и складываются на диск как WAV. В момент распознавания лица
    синтез не выполняется — отдаётся готовый файл, задержка минимальна.
    """

    def __init__(self, cache_dir="voice_cache", template="Здравствуйте, {name}!",
                 rate=None, voice_hint=None, synth_timeout=30.0):
        self.template = template
        self.cache_dir = os.path.join(cache_dir, "greetings")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.rate = rate
        self.voice_hint = voice_hint
        self.synth_timeout = synth_timeout
        self._queue = queue.Queue()
        self._lock = threading.Lock()
        self._pending = set()
        self._errors = {}
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="tts-worker")
        self._worker.start()

    def text_for(self, name):
        return format_template(self.template, name)

    def wav_path(self, name):
        digest = hashlib.sha1(self.text_for(name).encode("utf-8")).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{digest}.wav")

    def get(self, name):
        path = self.wav_path(name)
        try:
            if os.path.getsize(path) > 44:
                with open(path, "rb") as f:
                    return f.read()
        except OSError:
            pass
        return None

    def status(self, name):
        if self.get(name) is not None:
            return "ready"
        with self._lock:
            if name in self._pending:
                return "pending"
            if name in self._errors:
                return "error"
        return "missing"

    def ensure(self, name, force=False, wait=False, timeout=None):
        """Гарантировать наличие синтезированной фразы для имени."""
        if not force and self.status(name) == "ready":
            return "ready"
        deadline = time.time() + (timeout if timeout is not None else max(30.0, self.synth_timeout))
        with self._lock:
            already_pending = name in self._pending
            if not already_pending:
                self._pending.add(name)
                self._errors.pop(name, None)
        if already_pending:
            if wait:
                while self.status(name) == "pending" and time.time() < deadline:
                    time.sleep(0.05)
                return self.status(name)
            return "pending"
        ev = threading.Event() if wait else None
        self._queue.put((name, ev))
        if wait:
            ev.wait(timeout=max(0.0, deadline - time.time()))
        return self.status(name)

    def invalidate(self, name):
        with self._lock:
            self._pending.discard(name)
            self._errors.pop(name, None)
        try:
            os.remove(self.wav_path(name))
        except OSError:
            pass

    def _synthesize_in_subprocess(self, text, out_path):
        cmd = [sys.executable, "-c", TTS_SUBPROCESS_CODE, text, out_path,
               str(self.rate) if self.rate else "",
               self.voice_hint or ""]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=self.synth_timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"TTS процесс завершился с кодом {proc.returncode}: "
                               f"{(proc.stderr or '')[-300:]}")

    def _worker_loop(self):
        while not self._stop.is_set():
            try:
                name, ev = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            text = self.text_for(name)
            out_path = self.wav_path(name)
            tmp_path = out_path + ".part"
            ok = False
            last_err = None
            for _attempt in range(2):
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    self._synthesize_in_subprocess(text, tmp_path)
                    if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 44:
                        if os.path.exists(out_path):
                            os.remove(out_path)
                        os.replace(tmp_path, out_path)
                        ok = True
                        break
                    last_err = "TTS вернул пустой файл"
                    log.warning(f"{last_err} для {name}, повтор")
                except Exception as e:
                    last_err = e
                    log.error(f"Ошибка TTS для {name}: {e}")
            with self._lock:
                self._pending.discard(name)
                if not ok:
                    self._errors[name] = str(last_err) if last_err else "TTS вернул пустой файл"
            if ok:
                log.info(f"Озвучка готова: {name}")
            else:
                log.error(f"Не удалось синтезировать фразу для {name}")
            if ev:
                ev.set()
