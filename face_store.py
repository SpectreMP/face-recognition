# -*- coding: utf-8 -*-
import logging
import os
import re
import threading

import cv2
import numpy as np
import face_recognition

log = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
_SUFFIX_RE = re.compile(r"_\d+$")


def sanitize_name(name):
    name = re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", " ", str(name))
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name[:64].rstrip(" .")


class FaceStore:
    """Потокобезопасное хранилище эталонов: имя -> [(файл, эмбеддинг)]."""

    def __init__(self, faces_dir):
        self.faces_dir = faces_dir
        os.makedirs(faces_dir, exist_ok=True)
        self._lock = threading.RLock()
        self._persons = {}

    def load(self):
        persons = {}
        for fname in sorted(os.listdir(self.faces_dir)):
            if not fname.lower().endswith(IMAGE_EXTS):
                continue
            name = _SUFFIX_RE.sub("", os.path.splitext(fname)[0])
            if not name:
                continue
            path = os.path.join(self.faces_dir, fname)
            try:
                img = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(img)
            except Exception as e:
                log.error(f"Ошибка загрузки {fname}: {e}")
                continue
            if not encs:
                log.warning(f"В файле {fname} не найдено лиц")
                continue
            persons.setdefault(name, []).append((fname, np.asarray(encs[0])))
            log.info(f"Загружен эталон: {name} ({fname})")
        with self._lock:
            self._persons = persons
        total = sum(len(v) for v in persons.values())
        log.info(f"Всего эталонов: {total} ({len(persons)} чел.)")
        return dict(persons)

    def names(self):
        with self._lock:
            return sorted(self._persons.keys())

    def has(self, name):
        with self._lock:
            return name in self._persons

    def photos(self, name):
        with self._lock:
            return [f for f, _ in self._persons.get(name, [])]

    def entries(self):
        """Снимок [(имя, эмбеддинг)] для потока распознавания."""
        with self._lock:
            return [(n, enc.copy()) for n, lst in self._persons.items() for _, enc in lst]

    def _next_filename(self, safe):
        existing_lower = {f.lower() for f in os.listdir(self.faces_dir)}
        candidate = f"{safe}.jpg"
        idx = 1
        while candidate.lower() in existing_lower and idx < 99:
            idx += 1
            candidate = f"{safe}_{idx}.jpg"
        return candidate

    def add_photo(self, name, data):
        """Проверяет фото (наличие лица), сохраняет его и обновляет кэш.

        Возвращает (ok, message).
        """
        safe = sanitize_name(name)
        if not safe:
            return False, "Пустое или некорректное имя"
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return False, "Не удалось декодировать изображение"
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        if not boxes:
            return False, "На фото не найдено лицо"
        encs = face_recognition.face_encodings(rgb, boxes)
        enc = np.asarray(encs[0])
        ok_jpg, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok_jpg:
            return False, "Не удалось перекодировать изображение"
        with self._lock:
            fname = self._next_filename(safe)
            try:
                with open(os.path.join(self.faces_dir, fname), "wb") as f:
                    f.write(buf.tobytes())
            except OSError as e:
                log.error(f"Не удалось сохранить файл: {e}")
                return False, f"Ошибка записи файла: {e}"
            self._persons.setdefault(safe, []).append((fname, enc))
        log.info(f"Добавлен эталон: {safe} ({fname})")
        return True, fname

    def delete_person(self, name):
        with self._lock:
            lst = self._persons.pop(name, None)
        if not lst:
            return False
        for fname, _ in lst:
            try:
                os.remove(os.path.join(self.faces_dir, fname))
                log.info(f"Удалён файл эталона: {fname}")
            except OSError as e:
                log.error(f"Не удалось удалить {fname}: {e}")
        return True
