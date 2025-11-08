import cv2
import face_recognition
import pickle
import numpy as np
import mss
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODINGS_PATH = os.path.join(BASE_DIR, "encodings.pkl")

print("Carregando encodings...")
with open(ENCODINGS_PATH, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]
print(f"Encodings carregados para: {set(known_names)}")

sct = mss.mss()
monitor = sct.monitors[1]

print("O monitoramento começará em 5 segundos...")
time.sleep(5)
print("Monitorando sua tela em background. Pressione Ctrl+C para parar.\n")

while True:
    screenshot = np.array(sct.grab(monitor))
    rgb_frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        name = "Desconhecido"

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

        print(f"[DETECTADO] {name}")

    time.sleep(1)  # espera 1s entre capturas (ajusta pra aliviar CPU)
