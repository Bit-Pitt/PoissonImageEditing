'''
    Script per ridimensionare l'immagine "source.jpeg" alle dimensioni di "target.jpeg".
'''

import cv2

# Nomi delle immagini
source_name = "source.jpeg"
target_name = "target.avif"

# Carica immagini
source = cv2.imread(source_name)
target = cv2.imread(target_name)

if source is None:
    raise ValueError(f"Impossibile caricare {source_name}")

if target is None:
    raise ValueError(f"Impossibile caricare {target_name}")

# Dimensioni del target
height, width = target.shape[:2]

# Ridimensiona la source alle dimensioni del target
source_resized = cv2.resize(
    source,
    (width, height),
    interpolation=cv2.INTER_LINEAR
)

# Salva la nuova source
cv2.imwrite("source_resized.png", source_resized)

# Salva anche eventualmente il target con un nuovo nome
cv2.imwrite("target_resized.png", target)

print("Dimensioni target:", target.shape[:2])
print("Dimensioni source originale:", source.shape[:2])
print("Dimensioni source ridimensionata:", source_resized.shape[:2])