"""
Wrapper per ciascuna variante di Poisson Image Editing.

Ogni funzione ha la stessa firma (source, target, mask, **kwargs) -> result,
cosi' da poter essere registrata direttamente in main.py:METHODS. Eventuali
parametri specifici della variante (sigma, alpha, beta...) arrivano via
**kwargs dal parser CLI e vengono "fissati" nel guidance factory con
functools.partial prima di chiamare il motore generico solve_poisson.
"""
from functools import partial
import numpy as np

from skimage.color import rgb2gray
from skimage.feature import canny

from poisson import solve_poisson
from guidance import (
    cloning_guidance_factory,
    mixed_guidance_factory,
    flattening_guidance_factory,
    illumination_guidance_factory,
)

from tile_preprocessing import build_tileable_boundary_target, build_tiling_mask


def poisson_blend(source, target, mask, **kwargs):
    """Seamless cloning base: v = grad(source)."""
    return solve_poisson(source, target, mask, cloning_guidance_factory)


def mixed_gradients_blend(source, target, mask, **kwargs):
    """Mixed gradients: v = il gradiente con modulo maggiore tra source e target."""
    return solve_poisson(source, target, mask, mixed_guidance_factory)


def texture_flattening_blend(source, target, mask, sigma: float = 1.5, flatten_factor: float = 3.0, **kwargs):
    """Texture flattening: rileva i bordi nella sorgente con Canny
    (scikit-image) e azzera il gradiente altrove, appiattendo la texture
    nella regione clonata pur mantenendo i contorni principali.

    sigma: deviazione standard del filtro gaussiano usato da Canny prima
    della rilevazione dei bordi (valori piu' alti -> meno edge, piu' flat).
    """
    gray = rgb2gray(source / 255.0)
    edge_mask = canny(gray, sigma=sigma)
    factory = partial(flattening_guidance_factory, edge_mask=edge_mask, flatten_factor=flatten_factor)
    return solve_poisson(source, target, mask, factory)


def illumination_change_blend(source, target, mask, alpha: float = 0.2, beta: float = 0.2, **kwargs):
    """Local illumination change: comprime i gradienti forti mantenendone
    direzione e segno (vedi guidance.illumination_guidance_factory)."""
    factory = partial(illumination_guidance_factory, alpha=alpha, beta=beta)
    return solve_poisson(source, target, mask, factory)


from skimage.color import rgb2hsv, hsv2rgb
def local_color_change_blend(source, target, mask, hue_shift: float = 0.0, **kwargs):
    """Local color change via rotazione della tonalita' (hue) in spazio HSV.

    Non usiamo uno shift additivo costante in RGB: essendo costante su tutta
    Omega, si cancella identicamente nel gradiente interno (v_pq = T_p - T_q,
    invariato) e la soluzione esatta del sistema Poisson degenera nel
    copia-incolla puro (dimostrabile analiticamente). La rotazione hue invece
    e' non lineare e non uniforme in RGB (dipende da saturazione/luminosita'
    di ogni pixel), quindi introduce un vero gradiente che Poisson propaga
    con blending smussato al bordo.

    hue_shift: spostamento della tonalita' in [0,1], ciclico (es. 0.33 ~ rosso->verde).
    """
    target_hsv = rgb2hsv(target / 255.0)
    src_hsv = target_hsv.copy()
    src_hsv[..., 0] = (src_hsv[..., 0] + hue_shift) % 1.0
    synthetic_source = hsv2rgb(src_hsv) * 255.0

    return solve_poisson(synthetic_source, target, mask, cloning_guidance_factory)



def seamless_tiling_blend(source, target, mask, **kwargs):
    """Seamless tiling:  e' tutto nel preprocessing: si forza il bordo esterno a valori mediati (sx=dx,
    nord=sud) come condizione al contorno, e si lascia che Poisson propaghi
    la texture originale del tile verso quei nuovi bordi.

    source/target/mask passati vengono ignorati: tutto e' derivato da
    source stesso (che qui e' orig_tile).
    """
    tileable_target = build_tileable_boundary_target(source)
    tiling_mask = build_tiling_mask(source.shape)
    return solve_poisson(source, tileable_target, tiling_mask, cloning_guidance_factory)


from scipy.ndimage import binary_dilation       #trova i pixel appena fuori dalla maschera 
from scipy.ndimage import distance_transform_edt

def border_feathering_blend(source, target, mask, alpha: float = 0.5,
                              feather_width: int = 3, **kwargs):
    """Border feathering con alpha decrescente in funzione della distanza dal
    bordo: i pixel subito fuori dalla maschera prendono quasi tutto 'alpha'
    verso source, quelli più lontani (fino a feather_width) sfumano verso 0,
    cosi' la transizione e' morbida.

    Idea di fondo: la condizione al contorno di Dirichlet ancora Poisson
    esattamente al colore del target sul bordo. Se target e source hanno
    sfondi molto diversi, quell'ancoraggio e' "brusco" e trascina l'intera
    regione interna verso un colore molto lontano da quello originale della
    source . Sfumando il target verso la source proprio sul bordo, l'ancoraggio e' meno drastico
    e l'oggetto mantiene meglio il proprio colore originale.
    """
    # distanza (in pixel) di ogni punto FUORI mask dal pixel di mask piu' vicino
    dist_from_mask = distance_transform_edt(~mask)

    # anello di interesse: 1..feather_width pixel di distanza dal bordo
    boundary_ring = (dist_from_mask > 0) & (dist_from_mask <= feather_width)

    # decadimento lineare: alpha pieno a distanza 1, ~0 a distanza feather_width+1
    alpha_map = np.zeros(mask.shape, dtype=np.float64)
    alpha_map[boundary_ring] = alpha * (1.0 - (dist_from_mask[boundary_ring] - 1) / feather_width)
    alpha_map = alpha_map[:, :, None]  # broadcast sui 3 canali

    feathered_target = target.copy()
    feathered_target[boundary_ring] = (
        alpha_map[boundary_ring] * source[boundary_ring]
        + (1 - alpha_map[boundary_ring]) * target[boundary_ring]
    )

    return solve_poisson(source, feathered_target, mask, cloning_guidance_factory)

    