"""
Preproccesing per seamless tiling.
"""
import numpy as np


def build_tileable_boundary_target(tile: np.ndarray) -> np.ndarray:
    """Crea un'immagine target dove il bordo esterno (1 pixel) e' sostituito
    dalla media tra i bordi opposti (sx<->dx, nord<->sud), cosi' che
    affiancando piu' copie del tile i bordi combacino esattamente.
    """
    target = tile.copy().astype(np.float64)

    avg_h = (tile[:, 0, :] + tile[:, -1, :]) / 2.0   # media colonna sx/dx, per riga
    avg_v = (tile[0, :, :] + tile[-1, :, :]) / 2.0   # media riga nord/sud, per colonna

    target[:, 0, :] = avg_h
    target[:, -1, :] = avg_h
    target[0, :, :] = avg_v      # sovrascrive gli angoli con la media verticale
    target[-1, :, :] = avg_v

    return target


def build_tiling_mask(shape) -> np.ndarray:
    """Omega = tutta l'immagine TRANNE l'anello esterno di 1 pixel, che
    diventa la condizione al contorno (con i valori mediati sopra)."""
    H, W = shape[:2]
    mask = np.ones((H, W), dtype=bool)
    mask[0, :] = False
    mask[-1, :] = False
    mask[:, 0] = False
    mask[:, -1] = False
    return mask


def make_2x2_tile_preview(tile: np.ndarray) -> np.ndarray:
    """Affianca 2x2 copie dell'immagine per verificare visivamente la
    continuita' dei bordi dopo il seamless tiling."""
    top = np.concatenate([tile, tile], axis=1)
    return np.concatenate([top, top], axis=0)