import cv2
import face_recognition
import pickle
import numpy as np
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODINGS_PATH = os.path.join(BASE_DIR, "encodings.pkl")

# === 1. Carregar encodings ===
print("Carregando encodings...")
try:
    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)
    
    known_encodings = data["encodings"]
    known_names = data["names"]
    print(f"Encodings carregados para: {set(known_names)}")
except Exception as e:
    print(f"Erro ao carregar encodings: {e}")
    exit()


# === 2. Iniciar webcam ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # reduz resolução p/ aumentar FPS
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Iniciando monitoramento facial pela webcam...")
print("Pressione 'ESC' para sair")

fps_counter = 0
fps_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Reduz o frame para aumentar FPS (opcional)
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
        name = "Desconhecido"

        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

        face_names.append(name)

    # Volta posições para o tamanho original
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        color = (0, 255, 0) if name != "Desconhecido" else (0, 0, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # FPS
    fps_counter += 1
    if time.time() - fps_time >= 1:
        fps = fps_counter
        fps_counter = 0
        fps_time = time.time()
        cv2.putText(frame, f"FPS: {fps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Monitor de Reconhecimento Facial - Webcam", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
print("Monitoramento encerrado.")
