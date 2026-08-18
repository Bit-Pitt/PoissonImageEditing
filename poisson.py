"""
Poisson Image Editing - motore generico.

Il "motore" (build_A, build_b, solve) e' lo STESSO per tutte le varianti
(cloning, mixed gradients, texture flattening, illumination change).
Cio' che cambia tra una variante e l'altra e' solo il campo di guida v_pq,
vedi guidance.py.

Perche' A e' condivisa tra tutte le varianti:
A rappresenta il Laplaciano discreto sul dominio Omega (equazione 6 del
paper di Perez et al.) - dipende unicamente da QUALI pixel sono dentro/fuori
Omega, non dai valori dei gradienti. v_pq invece e' cio' che varia per
ottenere seamless cloning, mixed gradients, texture flattening, ecc.
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

    IDENTICA per tutte le varianti: dipende solo dalla topologia di Omega
    (quali pixel sono dentro, quali vicini sono anch'essi dentro), mai dai
    valori dei gradienti. Va costruita una sola volta per blend.
    """
    n = len(coords)
    H, W = mask.shape
    A = sp.lil_matrix((n, n), dtype=np.float64)

    for row_idx, (y, x) in enumerate(coords):
        A[row_idx, row_idx] = 4.0
        for ny, nx in _neighbors(y, x):
            if 0 <= ny < H and 0 <= nx < W and mask[ny, nx]:
                A[row_idx, index_map[ny, nx]] = -1.0

    return A.tocsr()


def build_b(target_channel: np.ndarray, mask: np.ndarray, coords, guidance_fn) -> np.ndarray:
    """Costruisce il vettore b per UN canale colore.

    guidance_fn(y, x, ny, nx) -> v_pq (scalare): il valore del campo di guida
    sull'edge p=(y,x) -> q=(ny,nx). E' l'UNICA cosa che cambia tra le varianti
    (vedi guidance.py per le diverse definizioni di v_pq).

    Per ogni pixel p in Omega:
      b[p] = somma di v_pq sui 4 vicini
           + somma dei valori noti di target per i vicini FUORI da Omega
             (condizioni al contorno di Dirichlet)
    """
    n = len(coords)
    H, W = mask.shape
    b = np.zeros(n, dtype=np.float64)

    for row_idx, (y, x) in enumerate(coords):
        val = 0.0
        for ny, nx in _neighbors(y, x):
            if 0 <= ny < H and 0 <= nx < W:
                val += guidance_fn(y, x, ny, nx)
                if not mask[ny, nx]:
                    val += target_channel[ny, nx]
            else:
                # vicino fuori dall'immagine: assumiamo v_pq=0 (gradiente nullo
                # al bordo dell'immagine) e trattiamo il "vicino fantasma" come
                # se avesse il valore target[p] stesso (bordo riflesso)
                val += target_channel[y, x]
        b[row_idx] = val

    return b


def solve_poisson(source: np.ndarray, target: np.ndarray, mask: np.ndarray, guidance_factory) -> np.ndarray:
    """Risolve il sistema di Poisson per tutti i canali, usando il campo di
    guida prodotto da guidance_factory.

    guidance_factory: callable con firma (channel_idx, source_ch, target_ch)
                       -> guidance_fn(y, x, ny, nx) -> v_pq
                       (vedi le factory in guidance.py)
    """
    assert source.shape == target.shape, "source e target devono avere la stessa shape"
    assert mask.shape == source.shape[:2], "mask deve avere la stessa H,W di source/target"

    index_map, coords = build_index_map(mask)
    if len(coords) == 0:
        raise ValueError("La maschera e' vuota: nessun pixel da clonare.")

    A = build_A(mask, index_map, coords)  # costruita UNA VOLTA, riusata per ogni canale

    result = target.copy()
    n_channels = source.shape[2]

    for c in range(n_channels):
        guidance_fn = guidance_factory(c, source[:, :, c], target[:, :, c])
        b = build_b(target[:, :, c], mask, coords, guidance_fn)
        x = spsolve(A, b)
        for idx, (y, xx) in enumerate(coords):
            result[y, xx, c] = x[idx]

    return result