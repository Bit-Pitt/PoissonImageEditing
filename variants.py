"""
Wrapper per ciascuna variante di Poisson Image Editing.

Ogni funzione ha la stessa firma (source, target, mask, **kwargs) -> result,
cosi' da poter essere registrata direttamente in main.py:METHODS. Eventuali
parametri specifici della variante (sigma, alpha, beta...) arrivano via
**kwargs dal parser CLI e vengono "fissati" nel guidance factory con
functools.partial prima di chiamare il motore generico solve_poisson.
"""
from functools import partial

from skimage.color import rgb2gray
from skimage.feature import canny

from poisson import solve_poisson
from guidance import (
    cloning_guidance_factory,
    mixed_guidance_factory,
    flattening_guidance_factory,
    illumination_guidance_factory,
)


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
