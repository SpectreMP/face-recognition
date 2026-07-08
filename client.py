import cv2
import zmq
import time
import threading
import json
import socket as sock
import logging
import argparse

# ---------- ПАРСИНГ АРГУМЕНТОВ ----------
parser = argparse.ArgumentParser(description="Клиент отправки видео на сервер")
parser.add_argument("--server-ip", type=str, default="127.0.0.1", help="IP-адрес сервера")
parser.add_argument("--video-port", type=int, default=5555, help="Порт для отправки видео (PUSH)")
parser.add_argument("--command-port", type=int, default=5556, help="Порт для получения команд (SUB)")
parser.add_argument("--camera-id", type=int, default=0, help="ID камеры (0, 1, ...)")
parser.add_argument("--fps", type=int, default=10, help="Кадров в секунду для отправки")
parser.add_argument("--client-id", type=str, default=sock.gethostname(), help="Идентификатор клиента")
args = parser.parse_args()

# ---------- НАСТРОЙКИ ----------
SERVER_IP = args.server_ip
VIDEO_PORT = args.video_port
COMMAND_PORT = args.command_port
CAMERA_ID = args.camera_id
SEND_INTERVAL = 1.0 / args.fps
CLIENT_ID = args.client_id

# ---------- ЛОГИРОВАНИЕ ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# ---------- ПОТОК ДЛЯ ПРИЁМА КОМАНД ----------
def command_listener():
    context = zmq.Context()
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect(f"tcp://{SERVER_IP}:{COMMAND_PORT}")
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, CLIENT_ID)
    log.info(f"Подписан на команды для {CLIENT_ID}")

    while True:
        try:
            client_id = sub_socket.recv_string()
            json_str = sub_socket.recv_string()
            msg = json.loads(json_str)
            log.info(f"Получена команда: {msg}")
            action = msg.get("action")
            if action == "greet":
                text = msg.get("text")
                log.info(f"Приветствие: {text}")
            # Здесь можно добавить другие действия
        except Exception as e:
            log.error(f"Ошибка в listener: {e}")
            time.sleep(0.1)

# ---------- ОСНОВНОЙ ПОТОК ОТПРАВКИ ВИДЕО ----------
def start_client():
    threading.Thread(target=command_listener, daemon=True).start()

    context = zmq.Context()
    video_socket = context.socket(zmq.PUSH)
    video_socket.connect(f"tcp://{SERVER_IP}:{VIDEO_PORT}")
    log.info(f"Подключен к серверу {SERVER_IP}:{VIDEO_PORT}")

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        log.error("Не удалось открыть камеру")
        return

    cv2.startWindowThread()
    log.info("Клиент запущен, отправка видео...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Client Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ret, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                video_socket.send_string(CLIENT_ID, flags=zmq.SNDMORE)
                video_socket.send(jpg.tobytes())

            time.sleep(SEND_INTERVAL)

    except KeyboardInterrupt:
        log.info("Клиент остановлен")
    finally:
        cap.release()
        video_socket.close()
        context.term()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    start_client()
