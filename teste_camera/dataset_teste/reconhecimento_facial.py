import os
import cv2
import face_recognition
import numpy as np

# 1. Definição dos Caminhos 📂
dataset_path = os.path.join("teste_camera", "dataset_teste")
imagePath1 = os.path.join(dataset_path, "andre.png")
imagePath2 = os.path.join(dataset_path, "duda.jpg")

# 2. Carregar as Imagens e Codificações Faciais 🧠
try:
    img_andre = face_recognition.load_image_file(imagePath1)
    img_duda = face_recognition.load_image_file(imagePath2)
    
    # Gerar as codificações (assumindo que há um rosto em cada imagem)
    enc_andre = face_recognition.face_encodings(img_andre)[0]
    enc_duda = face_recognition.face_encodings(img_duda)[0]

    # Armazenar rostos conhecidos
    known_encodings = [enc_andre, enc_duda]
    known_names = ["André", "Duda"]
except IndexError:
    print("ERRO: O face_recognition não encontrou um rosto em uma das imagens de treino. Verifique se as imagens estão visíveis.")
    exit()
except FileNotFoundError as e:
    print(f"ERRO: Arquivo não encontrado: {e}. Verifique se a pasta '{dataset_path}' está no mesmo diretório do script.")
    exit()

# 3. Inicializar a Webcam e Loop Principal 📹
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Erro ao acessar webcam. Verifique se a câmera está conectada/disponível.")
        break

    # Reduz o tamanho do frame para acelerar o processamento (1/4 do tamanho)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Converter de BGR (OpenCV) para RGB (face_recognition)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detectar rostos e gerar codificações no frame reduzido
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Desconhecido"

        # Escolher o rosto mais próximo (mesmo que não haja uma 'correspondência' perfeita)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        # Se a face mais próxima for uma correspondência dentro do limiar padrão
        if matches[best_match_index]:
            name = known_names[best_match_index]

        # Voltar coordenadas pro tamanho original (multiplicar por 4)
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Desenhar o retângulo e o nome
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

    # Mostrar resultado
    cv2.imshow("Reconhecimento Facial - André e Duda", frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 4. Finalização
video_capture.release()
cv2.destroyAllWindows()