import os
import cv2
import face_recognition
import pickle
from datetime import datetime

# Caminhos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
LANDMARKS_DIR = os.path.join(BASE_DIR, "landmarks")
OUTPUT_FILE = os.path.join(BASE_DIR, "encodings.pkl")

# Criar pasta de landmarks se não existir
os.makedirs(LANDMARKS_DIR, exist_ok=True)

encodings = []
names = []

print("\n📸 Iniciando geração de encodings e landmarks...\n")

# Percorre as pastas de pessoas
for person_name in os.listdir(DATASET_DIR):
    person_folder = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_folder):
        continue

    print(f"🧠 Processando: {person_name}")
    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        landmarks_list = face_recognition.face_landmarks(image)

        if len(face_encodings) == 0:
            print(f"⚠️ Nenhum rosto detectado em {img_name}, ignorando.")
            continue

        # Salvar encoding
        encodings.append(face_encodings[0])
        names.append(person_name)

        # Desenhar landmarks
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for landmarks in landmarks_list:
            for key, points in landmarks.items():
                for (x, y) in points:
                    cv2.circle(image_bgr, (x, y), 1, (0, 255, 0), -1)

        # Salvar imagem com landmarks
        output_path = os.path.join(LANDMARKS_DIR, f"{person_name}_{img_name}")
        cv2.imwrite(output_path, image_bgr)

print("\n💾 Salvando encodings em arquivo .pkl...")
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump({"encodings": encodings, "names": names}, f)

print(f"\n✅ Processo concluído! {len(encodings)} rostos foram codificados.")
print(f"Arquivo salvo em: {OUTPUT_FILE}")
print(f"Landmarks salvos em: {LANDMARKS_DIR}")
print(f"Data e hora: {datetime.now()}\n")
print(f" - {img_name}: {len(face_encodings)} rosto(s) codificado(s) e landmarks salvos.")