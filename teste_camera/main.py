import face_recognition
import cv2

# Carrega as imagens de treino
imagem_andre = face_recognition.load_image_file("dataset/andre.jpg")
imagem_duda = face_recognition.load_image_file("dataset/duda.jpg")

# Codifica os rostos (gera o vetor de 128 pontos)
cod_andre = face_recognition.face_encodings(imagem_andre)[0]
cod_duda = face_recognition.face_encodings(imagem_duda)[0]

rostos_conhecidos = [cod_andre, cod_duda]
nomes_conhecidos = ["André", "Duda"]

# Inicia a webcam
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Reduz o tamanho pra acelerar o processamento
    frame_pequeno = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_pequeno = frame_pequeno[:, :, ::-1]  # BGR -> RGB

    # Localiza rostos e faz encoding
    localizacoes = face_recognition.face_locations(rgb_pequeno)
    encodings = face_recognition.face_encodings(rgb_pequeno, localizacoes)

    for (top, right, bottom, left), encoding in zip(localizacoes, encodings):
        # Compara o rosto atual com os conhecidos
        comparacoes = face_recognition.compare_faces(rostos_conhecidos, encoding)
        nome = "Desconhecido"

        if True in comparacoes:
            indice = comparacoes.index(True)
            nome = nomes_conhecidos[indice]

        # Volta às coordenadas originais (porque reduzimos 0.25)
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Desenha o retângulo
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 255), cv2.FILLED)
        cv2.putText(frame, nome, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 2)

    cv2.imshow("Reconhecimento Facial", frame)

    # Tecla ESC pra sair
    if cv2.waitKey(1) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()
