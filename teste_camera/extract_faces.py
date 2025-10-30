# teste_camera/extract_faces.py
import os
import cv2
import face_recognition

dataset_path = os.path.join(os.path.dirname(__file__), "dataset_teste")
img_path = os.path.join(dataset_path, "duda.jpg")   # ou andre.jpg
out_dir = os.path.join(dataset_path, "rostos_detectados")
os.makedirs(out_dir, exist_ok=True)

img = face_recognition.load_image_file(img_path)
locations = face_recognition.face_locations(img)

for i, (top, right, bottom, left) in enumerate(locations):
    # recortar do array RGB
    face = img[top:bottom, left:right]
    face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
    save_path = os.path.join(out_dir, f"face_{i}.jpg")
    cv2.imwrite(save_path, face_bgr)
    print("Salvo:", save_path)

print("Concluído. Veja as imagens em:", out_dir)
