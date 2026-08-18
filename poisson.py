"""
Poisson Image Editing (Perez, Gangnet, Blake 2003) - seamless cloning base (v = grad(S)).

Pipeline:
  1. Si assegna un indice lineare a ogni pixel dentro la maschera Omega.
  2. Si costruisce la matrice sparsa A (Laplaciano discreto, 4-connesso).
  3. Si costruisce il vettore b (divergenza del campo di gradienti + boundary da target).
  4. Si risolve A x = b per ciascun canale colore separatamente.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def _neighbors(y, x):
    """I 4 vicini (4-connessi) di un pixel (y, x)."""
    return [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]


def build_index_map(mask: np.ndarray):
    """Assegna un indice lineare 0..N-1 a ogni pixel dentro la maschera.

    Ritorna:
      index_map: array (H, W) di int, -1 dove non c'e' un pixel Omega, altrimenti l'indice
      coords: lista di (y, x) nell'ordine corrispondente agli indici
    """
    ys, xs = np.nonzero(mask)
    coords = list(zip(ys.tolist(), xs.tolist()))
    index_map = -np.ones(mask.shape, dtype=np.int64)
    for idx, (y, x) in enumerate(coords):
        index_map[y, x] = idx
    return index_map, coords


def build_A(mask: np.ndarray, index_map: np.ndarray, coords) -> sp.csr_matrix:
    """Costruisce la matrice sparsa del Laplaciano discreto per i pixel in Omega.

    Per ogni pixel p in Omega:
      - 4 sulla diagonale (il numero di vicini, sempre 4 per grid 4-connessa)
      - -1 nella colonna di ogni vicino che e' ANCHE in Omega
      (i vicini fuori da Omega non generano una colonna: il loro contributo
       va nel vettore b, non in A - sono le condizioni al contorno di Dirichlet)
    """
    n = len(coords)
    H, W = mask.shape
    A = sp.lil_matrix((n, n), dtype=np.float64)

    for row_idx, (y, x) in enumerate(coords):
        A[row_idx, row_idx] = 4.0
        for ny, nx in _neighbors(y, x):
            if 0 <= ny < H and 0 <= nx < W and mask[ny, nx]:
                col_idx = index_map[ny, nx]
                A[row_idx, col_idx] = -1.0

    return A.tocsr()


def build_b(source: np.ndarray, target: np.ndarray, mask: np.ndarray, coords) -> np.ndarray:
    """Costruisce il vettore b per UN canale colore.

    Per ogni pixel p in Omega:
      b[p] = divergenza discreta di v (qui v = grad(source), quindi seamless cloning base)
           + somma dei valori noti di target per i vicini FUORI da Omega (Dirichlet boundary)
    """
    n = len(coords)
    H, W = mask.shape
    b = np.zeros(n, dtype=np.float64)

    for row_idx, (y, x) in enumerate(coords):
        # divergenza discreta del gradiente sorgente: 4*S(p) - somma S(vicini)
        val = 4.0 * source[y, x]
        for ny, nx in _neighbors(y, x):
            if 0 <= ny < H and 0 <= nx < W:
                val -= source[ny, nx]
                # se il vicino e' FUORI da Omega, il suo valore e' noto (= target)
                # e va aggiunto qui come termine noto (boundary condition)
                if not mask[ny, nx]:
                    val += target[ny, nx]
            else:
                # vicino fuori dall'immagine: trattalo come se valesse quanto il pixel stesso
                # (evita di dover gestire un caso limite raro se Omega tocca il bordo immagine)
                val += target[y, x]
        b[row_idx] = val

    return b


def poisson_blend(source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Esegue il seamless cloning base (v = grad(source)).

    source, target: array (H, W, 3) float64, stessa shape
    mask: array (H, W) bool

    Ritorna l'immagine risultato (H, W, 3) float64.
    """
    assert source.shape == target.shape, "source e target devono avere la stessa shape"
    assert mask.shape == source.shape[:2], "mask deve avere la stessa H,W di source/target"

    index_map, coords = build_index_map(mask)
    if len(coords) == 0:
        raise ValueError("La maschera e' vuota: nessun pixel da clonare.")

    A = build_A(mask, index_map, coords)

    result = target.copy()
    n_channels = source.shape[2]

    for c in range(n_channels):
        b = build_b(source[:, :, c], target[:, :, c], mask, coords)
        x = spsolve(A, b)
        for idx, (y, x_coord) in enumerate(coords):
            result[y, x_coord, c] = x[idx]

    return result
