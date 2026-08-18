"""
Utility di caricamento/salvataggio immagini per Poisson Image Editing.
"""
from PIL import Image
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Carica un'immagine RGB come array float64 in [0, 255], shape (H, W, 3)."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float64)


def load_mask(path: str) -> np.ndarray:
    """Carica una maschera come array booleano, shape (H, W).
    Qualsiasi pixel > 127 (dopo conversione a grayscale) e' considerato dentro Omega.
    """
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.uint8)
    return arr > 127


def save_image(arr: np.ndarray, path: str) -> None:
    """Salva un array float (range atteso [0,255]) come immagine RGB su disco."""
    clipped = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(clipped, mode="RGB").save(path)
