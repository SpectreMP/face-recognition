# -*- coding: utf-8 -*-
"""Проверка полного цикла без камеры: отправляет изображение как кадр видео
и слушает команды сервера. Пример:

    python e2e_check.py --image known_faces/michael.jpg --name tester
"""
import argparse
import json
import threading
import time

import cv2
import zmq


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="E2E проверка сервера распознавания")
    p.add_argument("--server-ip", default="127.0.0.1")
    p.add_argument("--video-port", type=int, default=5555)
    p.add_argument("--command-port", type=int, default=5556)
    p.add_argument("--client-id", default="e2e-check")
    p.add_argument("--image", required=True, help="JPEG/PNG файл с лицом")
    p.add_argument("--frames", type=int, default=3)
    p.add_argument("--fps", type=float, default=4.0)
    p.add_argument("--timeout", type=float, default=25.0, help="Сколько ждать команду greet")
    return p.parse_args(argv)


def main():
    args = parse_args()
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Не удалось прочитать {args.image}")
    h, w = img.shape[:2]
    if w > 800:
        scale = 800 / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise SystemExit("Не удалось закодировать кадр")

    ctx = zmq.Context()
    push = ctx.socket(zmq.PUSH)
    push.connect(f"tcp://{args.server_ip}:{args.video_port}")
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{args.server_ip}:{args.command_port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, args.client_id)

    got = {}

    def listener():
        while not got.get("stop"):
            if sub.poll(200):
                _topic = sub.recv_string()
                try:
                    msg = json.loads(sub.recv_string())
                except Exception as e:
                    print(f"[!] битый JSON: {e}")
                    continue
                audio_len = len(msg.get("audio_b64") or "")
                print(f"[<=] команда: action={msg.get('action')} name={msg.get('name')} "
                      f"text={msg.get('text')!r} audio_b64={audio_len} байт(base64)")
                if msg.get("action") == "greet":
                    got["msg"] = msg

    t = threading.Thread(target=listener, daemon=True)
    t.start()
    time.sleep(0.7)  # slow joiner

    interval = 1.0 / max(args.fps, 0.1)
    for i in range(args.frames):
        push.send_string(args.client_id, flags=zmq.SNDMORE)
        push.send(jpg.tobytes())
        print(f"[=>] кадр {i + 1}/{args.frames} отправен ({len(jpg)} байт)")
        time.sleep(interval)

    deadline = time.time() + args.timeout
    while time.time() < deadline and "msg" not in got:
        time.sleep(0.1)
    got["stop"] = True

    if "msg" in got:
        audio = got["msg"].get("audio_b64")
        print("OK: приветствие получено"
              + (f", аудио {len(audio)} b64-символов (~{len(audio) * 3 // 4} байт WAV)"
                 if audio else ", БЕЗ аудио"))
        return 0
    print("FAIL: приветствие не получено за отведённое время")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
