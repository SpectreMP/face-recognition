# -*- coding: utf-8 -*-
import argparse
import base64
import logging
import threading
import time

import cv2
import numpy as np
import zmq
import face_recognition

from face_store import FaceStore
from throttle import GreetThrottle
from voice_cache import VoiceCache, format_template
from webapp import create_app


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Сервер распознавания лиц")
    parser.add_argument("--video-port", type=int, default=5555, help="Порт для приёма видео (PULL)")
    parser.add_argument("--command-port", type=int, default=5556, help="Порт для отправки команд (PUB)")
    parser.add_argument("--known-faces-dir", type=str, default="known_faces", help="Папка с эталонными фото")
    parser.add_argument("--threshold", type=float, default=0.6, help="Порог схожести (0.0-1.0)")
    parser.add_argument("--no-debug", action="store_true", help="Отключить показ окна с видео")
    parser.add_argument("--save-debug", action="store_true", help="Сохранять кадры с аннотациями в папку debug_frames")
    parser.add_argument("--web-port", type=int, default=8080, help="Порт веб-панели управления")
    parser.add_argument("--no-web", action="store_true", help="Отключить веб-панель")
    parser.add_argument("--greet-cooldown", type=float, default=30.0,
                        help="Минимальный интервал между приветствиями одного человека, сек")
    parser.add_argument("--greet-template", type=str, default="Здравствуйте, {name}!",
                        help="Шаблон фразы приветствия ({name} — имя человека)")
    parser.add_argument("--voice-cache-dir", type=str, default="voice_cache",
                        help="Папка кэша синтезированных фраз")
    parser.add_argument("--tts-rate", type=int, default=None, help="Скорость речи TTS (слов в минуту)")
    parser.add_argument("--tts-voice", type=str, default=None,
                        help="Подстрока имени/ID голоса TTS (например, Irina)")
    return parser.parse_args(argv)


# ---------- ЛОГИРОВАНИЕ ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


# ---------- РАСПОЗНАВАНИЕ ----------
def recognize_face(entries, face_encoding, threshold):
    """entries — снимок из FaceStore.entries(): [(имя, эмбеддинг)]."""
    if not entries:
        return None, 1.0
    names = [n for n, _ in entries]
    matrix = np.stack([enc for _, enc in entries])
    distances = face_recognition.face_distance(matrix, face_encoding)
    best_idx = int(np.argmin(distances))
    if distances[best_idx] < threshold:
        return names[best_idx], float(distances[best_idx])
    return None, float(distances[best_idx])


# ---------- ОТПРАВКА КОМАНД ----------
def send_command(command_socket, client_id, payload):
    command_socket.send_string(client_id, flags=zmq.SNDMORE)
    command_socket.send_json(payload)


def build_greet_command(name, wav_bytes, template):
    payload = {
        "action": "greet",
        "name": name,
        "text": format_template(template, name),
        "audio_b64": None,
    }
    if wav_bytes:
        payload["audio_b64"] = base64.b64encode(wav_bytes).decode("ascii")
    return payload


# ---------- ГЛАВНЫЙ ЦИКЛ ----------
def start_server(args):
    show_debug = not args.no_debug
    save_debug = args.save_debug

    store = FaceStore(args.known_faces_dir)
    store.load()

    voice = VoiceCache(cache_dir=args.voice_cache_dir,
                       template=args.greet_template,
                       rate=args.tts_rate,
                       voice_hint=args.tts_voice)

    # Прогрев кэша: заранее синтезируем фразы для всех известных лиц
    for name in store.names():
        voice.ensure(name)

    throttle = GreetThrottle(args.greet_cooldown)

    context = zmq.Context()
    video_socket = context.socket(zmq.PULL)
    video_socket.bind(f"tcp://*:{args.video_port}")
    command_socket = context.socket(zmq.PUB)
    command_socket.bind(f"tcp://*:{args.command_port}")
    time.sleep(0.5)
    log.info(f"Сервер запущен. Приём видео на {args.video_port}, команды на {args.command_port}")

    if not args.no_web:
        app = create_app(store, voice, args.known_faces_dir)
        web_thread = threading.Thread(
            target=app.run,
            kwargs={"host": "0.0.0.0", "port": args.web_port,
                    "threaded": True, "debug": False, "use_reloader": False},
            daemon=True, name="web-panel")
        web_thread.start()
        log.info(f"Веб-панель: http://0.0.0.0:{args.web_port}")

    if save_debug:
        import os
        os.makedirs("debug_frames", exist_ok=True)

    frame_counter = 0
    if show_debug:
        try:
            cv2.startWindowThread()
        except Exception as e:
            log.warning(f"Не удалось запустить оконный поток: {e}")
            show_debug = False

    while True:
        try:
            # Неблокирующее чтение – берём только последний кадр
            last_msg = None
            while True:
                try:
                    msg = video_socket.recv_multipart(flags=zmq.NOBLOCK)
                    last_msg = msg
                except zmq.Again:
                    break
            if last_msg is None:
                time.sleep(0.001)
                continue

            client_id_bytes, jpg_bytes = last_msg[0], last_msg[1]
            client_id = client_id_bytes.decode('utf-8')
            log.debug(f"Получен кадр от {client_id}, размер {len(jpg_bytes)} байт")

            np_arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                log.warning("Не удалось декодировать кадр")
                continue

            frame_display = frame.copy()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            entries = store.entries()

            log.info(f"Обнаружено лиц: {len(face_locations)}")

            for (top, right, bottom, left), face_enc in zip(face_locations, face_encodings):
                name, distance = recognize_face(entries, face_enc, args.threshold)
                if name:
                    color = (0, 255, 0)
                    label = f"{name} ({distance:.3f})"
                    if throttle.allow(name):
                        wav_bytes = voice.get(name)
                        if wav_bytes is None:
                            # Кэша нет (например, фразу ещё не успели синтезировать) —
                            # отправляем текст без аудио и ставим синтез в очередь.
                            voice.ensure(name)
                            log.warning(f"Нет кэшированной озвучки для {name}, синтез запланирован")
                        send_command(command_socket, client_id,
                                     build_greet_command(name, wav_bytes, args.greet_template))
                        log.info(f"✅ Распознан {name} (dist={distance:.3f}), приветствие отправлено"
                                 + (" с аудио" if wav_bytes else " без аудио"))
                else:
                    color = (0, 0, 255)
                    label = f"Unknown ({distance:.3f})"
                    log.info(f"❌ Неизвестное лицо (dist={distance:.3f})")

                cv2.rectangle(frame_display, (left, top), (right, bottom), color, 2)
                cv2.putText(frame_display, label, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if show_debug:
                try:
                    cv2.imshow("Server Recognition", frame_display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s') and save_debug:
                        cv2.imwrite(f"debug_frames/frame_{frame_counter:06d}.jpg", frame_display)
                        log.info(f"Сохранён кадр {frame_counter}")
                except Exception as e:
                    log.error(f"Ошибка при показе окна: {e}. Отключаем визуализацию.")
                    show_debug = False
            elif frame_counter % 100 == 0:
                log.info(f"Обработано кадров: {frame_counter}")
            frame_counter += 1

        except KeyboardInterrupt:
            log.info("Остановка сервера по Ctrl+C")
            break
        except Exception as e:
            log.error(f"Ошибка в основном цикле: {e}")
            continue

    video_socket.close()
    command_socket.close()
    context.term()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_server(parse_args())
