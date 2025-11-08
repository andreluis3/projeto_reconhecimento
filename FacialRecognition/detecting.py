import cv2
import face_recognition
import pickle
import numpy as np
import mss
import os
import time

# === Caminho absoluto do arquivo encodings.pkl ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODINGS_PATH = os.path.join(BASE_DIR, "encodings.pkl")

# === 1. Carregar encodings ===
print("Carregando encodings...")
with open(ENCODINGS_PATH, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]
print(f"Encodings carregados para: {set(known_names)}")

# === 2. Inicializar captura de tela ===
sct = mss.mss()
monitor = sct.monitors[1]

print("O reconhecimento facial começará em 5 segundos...")
time.sleep(5)
print("Iniciando reconhecimento facial na tela. Pressione ESC para sair.")

# Cria janela persistente
cv2.namedWindow("Reconhecimento Facial na Tela", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Reconhecimento Facial na Tela", 800, 450)  # opcional: janela menor

while True:
    screenshot = np.array(sct.grab(monitor))
    rgb_frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Desconhecido"
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(screenshot, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(screenshot, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(screenshot, name, (left + 6, bottom - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    # Atualiza janela existente
    cv2.imshow("Reconhecimento Facial na Tela", screenshot)

    # Aguarda tecla ESC (27) para sair
    if cv2.waitKey(30) == 27:
        break

cv2.destroyAllWindows()
