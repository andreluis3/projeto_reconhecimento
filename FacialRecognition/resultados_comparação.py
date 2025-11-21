import os
import pickle
import cv2
import numpy as np
import face_recognition
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd

# ====== Caminhos ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODINGS_FILE = os.path.join(BASE_DIR, "encodings.pk")
TEST_DIR = os.path.join(BASE_DIR, "test_images")  # pasta com as imagens de teste

# ====== Carregar encodings ======
with open(ENCODINGS_FILE, 'rb') as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

# ====== Treinar classificador KNN ======
knn_clf = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn_clf.fit(known_encodings, known_names)

# ====== Avaliar no conjunto de teste ======
y_true = []
y_pred = []

# Classes conhecidas
classes_conhecidas = ["andre", "cr7", "luciano_hulk", "angelina_jolie"]

for person_name in os.listdir(TEST_DIR):
    person_path = os.path.join(TEST_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

    for file in os.listdir(person_path):
        img_path = os.path.join(person_path, file)
        image = cv2.imread(img_path)
        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, faces)

        if len(encodings) == 0:
            continue  # nenhuma face encontrada

        for enc in encodings:
            # Previsão do KNN
            distances, indices = knn_clf.kneighbors([enc], n_neighbors=1)
            min_distance = distances[0][0]

            if min_distance < 0.5:  # limiar ajustável
                name = knn_clf.predict([enc])[0]
            else:
                name = "desconhecido"

            # Caso o nome real não esteja entre as classes conhecidas,
            # marcamos o rótulo verdadeiro como "desconhecido"
            if person_name not in classes_conhecidas:
                y_true.append("desconhecido")
            else:
                y_true.append(person_name)

            y_pred.append(name)

# ====== Gerar métricas ======
labels = sorted(list(set(y_true + y_pred)))
acuracia = accuracy_score(y_true, y_pred)
matriz = confusion_matrix(y_true, y_pred, labels=labels)
relatorio = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
df_relatorio = pd.DataFrame(relatorio).transpose()

# ====== Exibir resultados ======
print("=== TABELA DE EFICIÊNCIA ===")
print(df_relatorio)
print(f"\nTaxa de acerto geral (Acurácia): {acuracia * 100:.2f}%")

# ====== Gráfico de métricas ======
metricas = df_relatorio.loc[[name for name in labels if name != "accuracy"], ['precision', 'recall', 'f1-score']]
metricas.plot(kind='bar', figsize=(9, 6))
plt.title('Eficiência por Classe (CNN + KNN)')
plt.ylabel('Valor')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# ====== Matriz de Confusão ======
plt.figure(figsize=(6,6))
plt.imshow(matriz, cmap='Blues')
plt.title('Matriz de Confusão')
plt.xticks(range(len(labels)), labels, rotation=45)
plt.yticks(range(len(labels)), labels)
plt.xlabel('Previsto')
plt.ylabel('Real')
for (i, j), val in np.ndenumerate(matriz):
    plt.text(j, i, val, ha='center', va='center', color='black')
plt.show()
