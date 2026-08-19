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

from preprocessing import build_tileable_boundary_target, build_tiling_mask


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
