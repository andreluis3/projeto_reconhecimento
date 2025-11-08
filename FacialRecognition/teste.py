import cv2
import numpy as np
import pyautogui
import time
import face_recognition
import pickle
import os 



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODINGS_PATH = os.path.join(BASE_DIR, "encodings.pkl")

print("Carregando encodings...")
with open(ENCODINGS_PATH, "rb") as f:
    data = pickle.load(f)

print("[INFO] Iniciando em 5 segundos... prepare-se.")
time.sleep(5)

print("[INFO] Monitorando tela. Pressione 'ESC' para sair.")

# Configurações de janela
cv2.namedWindow("Tela Monitorada", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tela Monitorada", 800, 450)

while True:
    # Capturar screenshot da tela
    screenshot = pyautogui.screenshot()
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Reduzir tamanho pra acelerar o reconhecimento
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detectar rostos
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Desconhecido"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Ajustar coordenadas para o tamanho original
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # Desenhar retângulo e nome
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Mostrar uma única janela
    cv2.imshow("Tela Monitorada", frame)

    # Pressione ESC para sair
    if cv2.waitKey(30) & 0xFF == 27:
        print("[INFO] Encerrando monitoramento...")
        break

cv2.destroyAllWindows()
