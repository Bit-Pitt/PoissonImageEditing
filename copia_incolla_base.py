from pathlib import Path
from PIL import Image
import numpy as np

# Trova la cartella dove si trova questo file .py e punta a data/test1
SCRIPT_DIR = Path(__file__).parent
SRC_PATH = SCRIPT_DIR / "data" / "test1"

offset = (0, 0)

# .resolve() mostra il percorso reale completo nel terminale in caso di errore
try:
    source = np.array(Image.open(SRC_PATH / "source.png"))
    target = np.array(Image.open(SRC_PATH / "target.png"))
    mask = np.array(Image.open(SRC_PATH / "mask.png").convert("L")) > 128
except FileNotFoundError as e:
    print(f"Errore! Controlla se il percorso esiste davvero: {SRC_PATH.resolve()}")
    raise e

oy, ox = offset
result = target.copy()
ys, xs = np.where(mask)
result[ys + oy, xs + ox] = source[ys, xs]

# Salva il risultato per verificare
Image.fromarray(result).save("testCopiaNaive.png")


