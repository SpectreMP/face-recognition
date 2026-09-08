# -*- coding: utf-8 -*-
import argparse
import base64
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import json
import socket as sock

import cv2
import zmq

# ---------- ЛОГИРОВАНИЕ ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

DEFAULT_PLAYERS = [
    ["paplay"],
    ["aplay", "-q"],
    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
    ["play", "-q"],
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Клиент отправки видео на сервер")
    parser.add_argument("--server-ip", type=str, default="127.0.0.1", help="IP-адрес сервера")
    parser.add_argument("--video-port", type=int, default=5555, help="Порт для отправки видео (PUSH)")
    parser.add_argument("--command-port", type=int, default=5556, help="Порт для получения команд (SUB)")
    parser.add_argument("--camera-id", type=int, default=0, help="ID камеры (0, 1, ...)")
    parser.add_argument("--fps", type=int, default=10, help="Кадров в секунду для отправки")
    parser.add_argument("--client-id", type=str, default=sock.gethostname(), help="Идентификатор клиента")
    parser.add_argument("--no-debug", action="store_true", help="Не показывать окно с камерой")
    parser.add_argument("--no-audio", action="store_true", help="Не проигрывать приветствия вслух")
    parser.add_argument("--audio-player", type=str, default=None,
                        help="Команда проигрывания WAV (по умолчанию авто: paplay/aplay/ffplay/play)")
    return parser.parse_args(argv)


def resolve_player(player_cmd=None):
    """Возвращает список аргументов плеера либо None, если плеер не найден."""
    if player_cmd:
        parts = player_cmd.split()
        if shutil.which(parts[0]):
            return parts
        log.warning(f"Указанный плеер не найден: {player_cmd}")
    for candidate in DEFAULT_PLAYERS:
        if shutil.which(candidate[0]):
            return candidate
    return None


def _cleanup_later(path, delay=30.0):
    def _rm():
        try:
            os.remove(path)
        except OSError:
            pass
    t = threading.Timer(delay, _rm)
    t.daemon = True
    t.start()


def play_wav_bytes(data, player_cmd=None):
    """Неблокирующее воспроизведение WAV из памяти через системный плеер.

    Возвращает True, если воспроизведение запущено.
    """
    player = resolve_player(player_cmd)
    if not player:
        log.error("Не найден аудиоплеер (paplay/aplay/ffplay/play). "
                  "Укажите --audio-player или установите pulseaudio-utils/alsa-utils")
        return False
    fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="greet_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
    except OSError as e:
        log.error(f"Не удалось записать временный WAV: {e}")
        return False
    cmd = player + [tmp_path]
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    except OSError as e:
        log.error(f"Не удалось запустить плеер {player[0]}: {e}")
        os.remove(tmp_path)
        return False
    _cleanup_later(tmp_path)
    return True


def handle_command(msg, play=True, player_cmd=None):
    """Обрабатывает одну команду от сервера. Возвращает описание действия."""
    action = msg.get("action")
    if action == "greet":
        text = msg.get("text") or f"Здравствуйте, {msg.get('name', 'гость')}!"
        log.info(f"Приветствие: {text}")
        audio_b64 = msg.get("audio_b64")
        played = False
        if audio_b64 and play:
            try:
                played = play_wav_bytes(base64.b64decode(audio_b64), player_cmd)
            except Exception as e:
                log.error(f"Ошибка воспроизведения: {e}")
        elif not audio_b64:
            log.info("Команда без аудио (озвучка ещё синтезируется)")
        return {"action": action, "name": msg.get("name"), "played": played}
    log.info(f"Получена команда: {msg}")
    return {"action": action}


# ---------- ПОТОК ДЛЯ ПРИЁМА КОМАНД ----------
def command_listener(args):
    context = zmq.Context()
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect(f"tcp://{args.server_ip}:{args.command_port}")
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, args.client_id)
    log.info(f"Подписан на команды для {args.client_id}")

    while True:
        try:
            _topic = sub_socket.recv_string()
            json_str = sub_socket.recv_string()
            msg = json.loads(json_str)
            handle_command(msg, play=not args.no_audio, player_cmd=args.audio_player)
        except Exception as e:
            log.error(f"Ошибка в listener: {e}")
            time.sleep(0.1)


# ---------- ОСНОВНОЙ ПОТОК ОТПРАВКИ ВИДЕО ----------
def start_client(args):
    threading.Thread(target=command_listener, args=(args,), daemon=True).start()

    context = zmq.Context()
    video_socket = context.socket(zmq.PUSH)
    video_socket.connect(f"tcp://{args.server_ip}:{args.video_port}")
    log.info(f"Подключен к серверу {args.server_ip}:{args.video_port}")

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        log.error("Не удалось открыть камеру")
        return

    send_interval = 1.0 / max(args.fps, 1)
    show_debug = not args.no_debug
    log.info("Клиент запущен, отправка видео...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if show_debug:
                try:
                    cv2.imshow("Client Camera", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception as e:
                    log.error(f"Ошибка окна: {e}. Отключаем визуализацию.")
                    show_debug = False

            ret, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                video_socket.send_string(args.client_id, flags=zmq.SNDMORE)
                video_socket.send(jpg.tobytes())

            time.sleep(send_interval)

    except KeyboardInterrupt:
        log.info("Клиент остановлен")
    finally:
        cap.release()
        video_socket.close()
        context.term()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    start_client(parse_args())
