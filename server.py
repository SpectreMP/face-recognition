import cv2
import numpy as np
import zmq
import os
import time
import logging
import face_recognition
import argparse

# ---------- ПАРСИНГ АРГУМЕНТОВ ----------
parser = argparse.ArgumentParser(description="Сервер распознавания лиц")
parser.add_argument("--video-port", type=int, default=5555, help="Порт для приёма видео (PULL)")
parser.add_argument("--command-port", type=int, default=5556, help="Порт для отправки команд (PUB)")
parser.add_argument("--known-faces-dir", type=str, default="known_faces", help="Папка с эталонными фото")
parser.add_argument("--threshold", type=float, default=0.6, help="Порог схожести (0.0-1.0)")
parser.add_argument("--no-debug", action="store_true", help="Отключить показ окна с видео")
parser.add_argument("--save-debug", action="store_true", help="Сохранять кадры с аннотациями в папку debug_frames")
args = parser.parse_args()

# ---------- НАСТРОЙКИ ИЗ АРГУМЕНТОВ ----------
VIDEO_PORT = args.video_port
COMMAND_PORT = args.command_port
KNOWN_FACES_DIR = args.known_faces_dir
MATCH_THRESHOLD = args.threshold
SHOW_DEBUG = not args.no_debug
SAVE_DEBUG_FRAMES = args.save_debug

# ---------- ЛОГИРОВАНИЕ ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# ---------- ЗАГРУЗКА ЭТАЛОНОВ ----------
def load_known_faces():
    known_encodings = {}
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        log.warning(f"Папка {KNOWN_FACES_DIR} создана, поместите туда эталонные фото")
        return known_encodings
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(filename)[0]
            path = os.path.join(KNOWN_FACES_DIR, filename)
            try:
                img = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(img)
                if len(encodings) == 0:
                    log.warning(f"В файле {filename} не найдено лиц")
                    continue
                known_encodings[name] = encodings[0]
                log.info(f"Загружен эталон: {name}")
            except Exception as e:
                log.error(f"Ошибка загрузки {filename}: {e}")
    return known_encodings

known_encodings = load_known_faces()
log.info(f"Всего эталонов: {len(known_encodings)}")

# ---------- РАСПОЗНАВАНИЕ ----------
def recognize_face(face_encoding):
    if not known_encodings:
        return None, 1.0
    distances = []
    names = []
    for name, known_enc in known_encodings.items():
        dist = face_recognition.face_distance([known_enc], face_encoding)[0]
        distances.append(dist)
        names.append(name)
    best_idx = np.argmin(distances)
    if distances[best_idx] < MATCH_THRESHOLD:
        return names[best_idx], distances[best_idx]
    else:
        return None, distances[best_idx]

# ---------- ОТПРАВКА КОМАНД ----------
def send_command(command_socket, client_id, command_data):
    command_socket.send_string(client_id, flags=zmq.SNDMORE)
    command_socket.send_json(command_data)

# ---------- ГЛАВНЫЙ ЦИКЛ ----------
def start_server():
    show_debug = SHOW_DEBUG
    context = zmq.Context()
    video_socket = context.socket(zmq.PULL)
    video_socket.bind(f"tcp://*:{VIDEO_PORT}")
    command_socket = context.socket(zmq.PUB)
    command_socket.bind(f"tcp://*:{COMMAND_PORT}")
    time.sleep(0.5)
    log.info(f"Сервер запущен. Приём видео на {VIDEO_PORT}, команды на {COMMAND_PORT}")

    if SAVE_DEBUG_FRAMES and not os.path.exists("debug_frames"):
        os.makedirs("debug_frames")

    frame_counter = 0
    cv2.startWindowThread()

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

            log.info(f"Обнаружено лиц: {len(face_locations)}")

            for (top, right, bottom, left), face_enc in zip(face_locations, face_encodings):
                name, distance = recognize_face(face_enc)
                if name:
                    color = (0, 255, 0)
                    label = f"{name} ({distance:.3f})"
                    send_command(command_socket, client_id, {
                        "action": "greet",
                        "name": name,
                        "text": f"Здравствуйте, {name}!"
                    })
                    log.info(f"✅ Распознан {name} (dist={distance:.3f})")
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
                    elif key == ord('s') and SAVE_DEBUG_FRAMES:
                        cv2.imwrite(f"debug_frames/frame_{frame_counter:06d}.jpg", frame_display)
                        log.info(f"Сохранён кадр {frame_counter}")
                        frame_counter += 1
                except Exception as e:
                    log.error(f"Ошибка при показе окна: {e}. Отключаем визуализацию.")
                    show_debug = False
            else:
                if frame_counter % 10 == 0:
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
    start_server()
