
import os
import face_recognition as fr
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont 

# 1. Definição dos Caminhos 📂
# ATENÇÃO: Verifique se estes caminhos estão corretos com base na sua estrutura de pastas.
DATASET_PATH = "teste_camera/dataset_teste"
IMG_ANDRE_PATH = os.path.join(DATASET_PATH, "andre.png")
# Usaremos 'duda.jpg' como a imagem de TESTE, pois ela deve ter 2 rostos.
IMG_DUDA_TESTE_PATH = os.path.join(DATASET_PATH, "duda.jpg") 


known_encodings = []
known_names = []

print("1. Codificando rostos conhecidos...")

try:
    # --- Rosto de André (Imagem de treino) ---
    img_andre_treino = fr.load_image_file(IMG_ANDRE_PATH)
    # [0] garante que pegamos a codificação do primeiro rosto detectado
    enc_andre = fr.face_encodings(img_andre_treino)[0] 
    known_encodings.append(enc_andre)
    known_names.append("Andre")
    
    
    img_duda_treino = fr.load_image_file(IMG_DUDA_TESTE_PATH)
    all_duda_encodings = fr.face_encodings(img_duda_treino)

    
    if len(all_duda_encodings) > 1:
        enc_duda = all_duda_encodings[1] 
        known_encodings.append(enc_duda)
        known_names.append("Duda")
    else:
         print("AVISO: A imagem de Duda não tem múltiplos rostos. Codificando o único rosto como Duda.")
         enc_duda = all_duda_encodings[0]
         known_encodings.append(enc_duda)
         known_names.append("Duda")


    print(f"Rostos conhecidos codificados: {known_names}")

except IndexError:
    print("ERRO: Um rosto não pôde ser encontrado em uma das imagens de treino. Verifique a visibilidade.")
    exit()
except FileNotFoundError as e:
    print(f"ERRO: Arquivo de treino não encontrado: {e}.")
    exit()


print("\n2. Processando imagem de teste (duda.jpg)...")

# Carrega a imagem de teste como array RGB
img_teste_rgb = fr.load_image_file(IMG_DUDA_TESTE_PATH)

# Encontra a localização e codificações de todos os rostos na imagem de teste
locations = fr.face_locations(img_teste_rgb)
encodings = fr.face_encodings(img_teste_rgb, locations)

# Converter o array RGB para um objeto PIL Image para desenho e exibição
pil_image = Image.fromarray(img_teste_rgb)
draw = ImageDraw.Draw(pil_image)

# Configuração de fonte (se quiser usar uma fonte específica)
try:
    font = ImageFont.truetype("arial.ttf", 25) 
except IOError:
    font = ImageFont.load_default()

# 4. COMPARAR E IDENTIFICAR ROSTOS
face_names = []
for (top, right, bottom, left), face_encoding in zip(locations, encodings):
    matches = fr.compare_faces(known_encodings, face_encoding)
    name = "Desconhecido"

    # Escolher o rosto mais próximo
    face_distances = fr.face_distance(known_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    
    if matches[best_match_index]:
        name = known_names[best_match_index]
        print(f"Rosto detectado: {name} (Distância: {face_distances[best_match_index]:.2f})")
    else:
        print(f"Rosto detectado: Desconhecido (Distância: {face_distances[best_match_index]:.2f})")

    face_names.append(name)
    
    # Desenhar o retângulo usando Pillow (coordenadas: left, top, right, bottom)
    draw.rectangle([left, top, right, bottom], outline=(0, 255, 0), width=3)
    
    # Desenhar a caixa de nome - CÓDIGO CORRIGIDO AQUI:
    
    # Usa textbbox para obter as dimensões da caixa de texto (left, top, right, bottom)
    bbox = draw.textbbox((0, 0), name, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Desenhar a caixa de nome
    draw.rectangle([left, bottom - text_height - 10, right, bottom], fill=(0, 255, 0), outline=(0, 255, 0))
    draw.text((left + 6, bottom - text_height - 5), name, font=font, fill=(255, 255, 255))
    
pil_image.show()

print("\nReconhecimento concluído. A imagem deve ter sido aberta em uma nova janela.")
