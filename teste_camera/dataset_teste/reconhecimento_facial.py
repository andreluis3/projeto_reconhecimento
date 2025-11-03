import os
import cv2
import face_recognition
import numpy as np
import time # Importado para pausar um pouco a identificação (opcional)

# 1. Definição dos Caminhos 📂
dataset_path = os.path.join("teste_camera", "dataset_teste")
imagePath1 = os.path.join(dataset_path, "andre.png")
imagePath2 = os.path.join(dataset_path, "duda.jpg")

# --- Variáveis de Estado (Para otimização) ---
face_locations = []
face_encodings = []
process_this_frame = True # Apenas para processar a cada dois frames (opcional)

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
    print(f"ERRO: Arquivo não encontrado: {e}. Verifique se a pasta '{dataset_path}' está no diretório correto.")
    exit()

# 3. Inicializar a Webcam e Loop Principal 📹
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Erro ao acessar webcam. Verifique se a câmera está conectada/disponível.")
        time.sleep(1) # Pausa antes de tentar novamente ou sair
        break

    # Processa o reconhecimento apenas a cada dois frames para maior velocidade
    if process_this_frame:
        # Reduz o tamanho do frame para acelerar o processamento (1/4 do tamanho)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Converter de BGR (OpenCV) para RGB (face_recognition)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detectar rostos e gerar codificações no frame reduzido
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        # Lista para armazenar os nomes identificados no frame atual
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Desconhecido"

            # Escolher o rosto mais próximo
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                name = known_names[best_match_index]

            face_names.append(name)
            
    process_this_frame = not process_this_frame


    # Desenhar as caixas e textos no frame original
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Voltar coordenadas pro tamanho original (multiplicar por 4)
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        # 🎨 Definir Cores e Texto de Exibição
        if name == "André":
            display_text = "Oi André"
            box_color = (0, 255, 0) # Verde
        elif name == "Duda":
            display_text = "Oi Duda"
            box_color = (0, 255, 0) # Verde
        else:
            display_text = "?"
            box_color = (0, 0, 255) # Vermelho (para Desconhecido)

        # Desenhar o retângulo e o nome
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
        cv2.putText(frame, display_text, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

    # Mostrar resultado
    cv2.imshow("Reconhecimento Facial - André e Duda", frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 4. Finalização
video_capture.release()
cv2.destroyAllWindows()