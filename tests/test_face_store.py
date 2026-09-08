# -*- coding: utf-8 -*-
import os

import cv2
import numpy as np

from face_store import FaceStore, sanitize_name


def png_bytes(w=40, h=40):
    ok, buf = cv2.imencode(".png", np.zeros((h, w, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


def test_sanitize_name():
    assert sanitize_name("  Иван   Петров ") == "Иван Петров"
    assert sanitize_name("../evil/name") == "evil name"
    assert sanitize_name('a<b>:"|?*\x00') == "a b"
    assert sanitize_name("") == ""
    assert len(sanitize_name("x" * 200)) == 64


def test_add_photo_creates_file_and_entry(fake_fr, tmp_path):
    store = FaceStore(str(tmp_path))
    ok, msg = store.add_photo("Михаил", png_bytes())
    assert ok is True
    assert msg == "Михаил.jpg"
    assert (tmp_path / "Михаил.jpg").exists()
    assert store.has("Михаил")
    assert store.photos("Михаил") == ["Михаил.jpg"]
    entries = store.entries()
    assert len(entries) == 1
    name, enc = entries[0]
    assert name == "Михаил"
    assert enc.shape == (128,)


def test_add_second_photo_gets_suffix(fake_fr, tmp_path):
    store = FaceStore(str(tmp_path))
    ok1, f1 = store.add_photo("Анна", png_bytes())
    ok2, f2 = store.add_photo("Анна", png_bytes())
    assert ok1 and ok2
    assert f1 == "Анна.jpg"
    assert f2 == "Анна_2.jpg"
    assert sorted(store.photos("Анна")) == ["Анна.jpg", "Анна_2.jpg"]
    assert len(store.entries()) == 2


def test_add_photo_without_face_rejected(fake_fr, tmp_path):
    fake_fr.faces_present = False
    store = FaceStore(str(tmp_path))
    ok, msg = store.add_photo("Боб", png_bytes())
    assert ok is False
    assert "лицо" in msg.lower()
    assert not list(tmp_path.iterdir())


def test_add_garbage_bytes_rejected(fake_fr, tmp_path):
    store = FaceStore(str(tmp_path))
    ok, msg = store.add_photo("Боб", b"\xff\xffnotanimage")
    assert ok is False
    assert "декодировать" in msg


def test_delete_person_removes_files(fake_fr, tmp_path):
    store = FaceStore(str(tmp_path))
    store.add_photo("Анна", png_bytes())
    store.add_photo("Анна", png_bytes())
    files = [tmp_path / f for f in ("Анна.jpg", "Анна_2.jpg")]
    assert all(f.exists() for f in files)
    assert store.delete_person("Анна") is True
    assert not store.has("Анна")
    assert not any(f.exists() for f in files)
    assert store.delete_person("Анна") is False


def test_load_groups_suffixes(fake_fr, tmp_path):
    for fname in ("michael.jpg", "michael_2.jpg", "anna.png", "readme.txt"):
        (tmp_path / fname).write_bytes(b"stub")
    store = FaceStore(str(tmp_path))
    persons = store.load()
    assert sorted(persons.keys()) == ["anna", "michael"]
    assert len(persons["michael"]) == 2
    assert store.names() == ["anna", "michael"]


def test_entries_snapshot_is_isolated(fake_fr, tmp_path):
    store = FaceStore(str(tmp_path))
    store.add_photo("Анна", png_bytes())
    snap = store.entries()
    snap[0] = ("Взломщик", snap[0][1])
    assert store.entries()[0][0] == "Анна"
