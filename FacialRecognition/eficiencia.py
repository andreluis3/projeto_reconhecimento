# eficiencia.py
import os
import pickle
import face_recognition
from typing import List, Tuple
import sys
print("PYTHON EM USO:", sys.executable)



import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")


print("Procurando dataset em:", dataset_dir)
print("Existe?", os.path.exists(dataset_dir))


def carregar_caminhos_e_rotulos(dataset_dir: str = "dataset") -> Tuple[List[str], List[str]]:
    caminhos = []
    rotulos = []

    ignorar = {"landmarks", "modelos", ".DS_Store", "__pycache__"}

    for nome_pasta in sorted(os.listdir(dataset_dir)):
        caminho_pasta = os.path.join(dataset_dir, nome_pasta)

        if not os.path.isdir(caminho_pasta):
            continue
        if nome_pasta in ignorar:
            continue

        for arquivo in sorted(os.listdir(caminho_pasta)):
            if arquivo.lower().endswith((".jpg", ".jpeg", ".png")):
                caminhos.append(os.path.join(caminho_pasta, arquivo))
                rotulos.append(nome_pasta)

    return caminhos, rotulos


def carregar_encodings(encodings_path: str = "encodings.pkl"):
    if not os.path.exists(encodings_path):
        raise FileNotFoundError(f"Arquivo de encodings não encontrado em: {encodings_path}")

    with open(encodings_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and 'encodings' in data and 'names' in data:
        known_encodings = data['encodings']
        known_names = data['names']
        return known_encodings, known_names

    # Forma 2: lista de tuplas (name, encoding)
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], (list, tuple)) and len(data[0]) == 2:
        known_names = []
        known_encodings = []
        for name, enc in data:
            known_names.append(name)
            known_encodings.append(enc)
        return known_encodings, known_names

    # Forma 3: dict com nomes como chaves e encoding como valor
    if isinstance(data, dict):
        # tenta interpretar cada chave como nome e valor como encoding
        possible_names = []
        possible_encs = []
        for k, v in data.items():
            if isinstance(v, (list, tuple)):
                possible_names.append(str(k))
                possible_encs.append(v)
        if possible_encs:
            return possible_encs, possible_names

    raise ValueError("Formato de encodings.pkl não reconhecido. Estrutura esperada: dict{'encodings','names'} ou lista de (name,enc).")


def prever_nome_para_imagem(image_path: str, known_encodings, known_names, tolerance: float = 0.45) -> str:
    """
    Tenta extrair o encoding da imagem e comparar com os encodings conhecidos.
    Retorna o nome com menor distância se estiver abaixo do tolerance, caso contrário 'unknown'.
    """
    if face_recognition is None:
        raise RuntimeError("face_recognition não disponível. Instale-a ou adapte a função de previsão ao seu pipeline.")

    image = face_recognition.load_image_file(image_path)
    encs = face_recognition.face_encodings(image)

    if len(encs) == 0:
        return "unknown"  # sem rosto detectado

    # se houver múltiplos rostos pega o primeiro
    encoding = encs[0]

    # calcula distâncias e pega o menor
    distances = face_recognition.face_distance(known_encodings, encoding)
    if len(distances) == 0:
        return "unknown"

    best_idx = int(distances.argmin())
    best_distance = distances[best_idx]
    best_name = known_names[best_idx]

    if best_distance <= tolerance:
        return best_name
    else:
        return "unknown"


def gerar_metricas(y_true: List[str], y_pred: List[str]) -> Tuple[pd.DataFrame, List[List[int]]]:
    acuracia = accuracy_score(y_true, y_pred)
    precisao = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    revocacao = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    matriz = confusion_matrix(y_true, y_pred, labels=sorted(list(set(y_true + y_pred))))

    tabela = pd.DataFrame({
        'Acurácia': [acuracia],
        'Precisão': [precisao],
        'Revocação (Recall)': [revocacao],
        'F1-Score': [f1]
    })

    return tabela, matriz


def main():
    base_dir = os.path.dirname(__file__)
    dataset_dir = os.path.join(base_dir, "dataset")
    base_dir = os.path.dirname(__file__)
    encodings_path = os.path.join(base_dir, "encodings.pkl")
    tolerance = 0.45                 # experimente 0.4 ~ 0.6 para ajustar sensibilidade

    print("Carregando caminhos e rótulos do dataset...")
    caminhos, rotulos = carregar_caminhos_e_rotulos(dataset_dir)
    print(f"Imagens encontradas: {len(caminhos)}. Pessoas (rótulos) únicas: {len(set(rotulos))}.")

    print("Carregando encodings conhecidos...")
    known_encodings, known_names = carregar_encodings(encodings_path)
    print(f"Encodings carregados: {len(known_encodings)} para {len(set(known_names))} nomes únicos.")

    y_true = rotulos
    y_pred = []

    print("Iniciando previsões nas imagens (isso pode demorar dependendo do tamanho)...")
    for i, caminho in enumerate(caminhos, 1):
        try:
            pred = prever_nome_para_imagem(caminho, known_encodings, known_names, tolerance=tolerance)
        except Exception as e:
            pred = "unknown"
            print(f"[ERRO] ao prever {caminho}: {e}")
        y_pred.append(pred)
        print(f"{i}/{len(caminhos)}  -> {os.path.basename(caminho)}  true={y_true[i-1]}  pred={pred}")

    # calcular métricas
    tabela_metricas, matriz_confusao = gerar_metricas(y_true, y_pred)

    print("\nTABELA DE MÉTRICAS:")
    print(tabela_metricas.to_string(index=False))

    # preparar labels ordenadas para a matriz
    labels = sorted(list(set(y_true + y_pred)))
    df_matriz = pd.DataFrame(matriz_confusao, index=labels, columns=labels)

    print("\nMATRIZ DE CONFUSÃO:")
    print(df_matriz)

    import pickle
    with open(encodings_path, "rb") as f:
        data = pickle.load(f)
    print(len(data["encodings"]))
    print(set(data["names"]))



    tabela_metricas.to_csv("metricas_eficiencia.csv", index=False)
    df_matriz.to_csv("matriz_confusao.csv", index=True)
    print("\nResultados salvos: metricas_eficiencia.csv, matriz_confusao.csv")




if __name__ == "__main__":
    main()
