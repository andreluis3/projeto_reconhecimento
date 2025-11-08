# validate_dataset.py
import os

BASE = os.path.dirname(__file__)
DATASET = os.path.join(BASE, "dataset")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

if not os.path.exists(DATASET):
    print("ERRO: pasta 'dataset' não encontrada em:", DATASET)
    raise SystemExit(1)

people = [d for d in os.listdir(DATASET) if os.path.isdir(os.path.join(DATASET, d))]
if not people:
    print("AVISO: não há subpastas em 'dataset'. Cada pessoa precisa de sua pasta com imagens.")
    raise SystemExit(0)

print(f"Encontradas {len(people)} pastas de pessoas no dataset.\n")
total_images = 0
for p in sorted(people):
    p_path = os.path.join(DATASET, p)
    imgs = [f for f in os.listdir(p_path) if f.lower().endswith(IMAGE_EXTS)]
    nonimgs = [f for f in os.listdir(p_path) if not f.lower().endswith(IMAGE_EXTS)]
    print(f" - {p}: {len(imgs)} imagens", end="")
    if nonimgs:
        print(f"  [arquivos não-imagem: {len(nonimgs)} -> {nonimgs[:5]}{'...' if len(nonimgs)>5 else ''}]", end="")
    if len(imgs) == 0:
        print("  <<< PASTA VAZIA! >>>", end="")
    print()
    total_images += len(imgs)
print(f"\nTotal de imagens encontradas: {total_images}")
print("\nChecar nomes: use CamelCase sem espaços (ex: CristianoRonaldo ou Cr7).")
