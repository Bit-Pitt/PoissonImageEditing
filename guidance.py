"""
Campi di guida (v) per le diverse varianti di Poisson Image Editing.

Ogni *_guidance_factory ha firma (channel_idx, source_ch, target_ch) -> guidance_fn,
dove guidance_fn(y, x, ny, nx) -> v_pq (scalare). Questa firma e' quella
richiesta da poisson.solve_poisson.

Se una variante ha bisogno di parametri extra (soglie, edge map precalcolate...),
si usa functools.partial per "fissarli" prima di passare la factory a
solve_poisson (vedi variants.py per gli esempi).
"""
import numpy as np


def cloning_guidance_factory(channel_idx, source_ch, target_ch):
    """Seamless cloning base (Perez et al., sez 3): v_pq = S_p - S_q.
    Preserva interamente la struttura della sorgente."""
    def v(y, x, ny, nx):
        return source_ch[y, x] - source_ch[ny, nx]
    return v


def mixed_guidance_factory(channel_idx, source_ch, target_ch):
    """Mixed gradients (Perez et al., sez 3.2): tra i due gradienti (source
    e target) su un edge, si tiene quello con modulo maggiore.
    Utile quando si vuole far "riemergere" texture della destinazione sotto
    la regione clonata (es. sorgente semi-trasparente, testo su texture)."""
    def v(y, x, ny, nx):
        d_s = source_ch[y, x] - source_ch[ny, nx]
        d_t = target_ch[y, x] - target_ch[ny, nx]
        return d_s if abs(d_s) > abs(d_t) else d_t
    return v


def flattening_guidance_factory(channel_idx, source_ch, target_ch, edge_mask, flatten_factor=3.0):
    """Texture flattening (Perez et al., sez 3.3): si mantiene il gradiente
    originale SOLO in corrispondenza di edge rilevati (edge_mask, es. da
    skimage.feature.canny), altrove si azzera -> appiattisce la texture
    pur mantenendo i contorni principali dell'oggetto.

    edge_mask: array booleano (H, W), True dove c'e' un edge rilevato.
    """
    def v(y, x, ny, nx):
        if edge_mask[y, x] or edge_mask[ny, nx]:
            return source_ch[y, x] - source_ch[ny, nx]
        else:
            return (source_ch[y, x] - source_ch[ny, nx]) / flatten_factor  # attenua il gradiente altrove
    return v


def illumination_guidance_factory(channel_idx, source_ch, target_ch, alpha=0.2, beta=0.2):
    """Local illumination change (Perez et al., sez 3.4): comprime la
    magnitudine dei gradienti forti mantenendone la direzione e il segno,
    cosi' da attenuare grosse variazioni di intensita' (es. ombre nette)
    preservando i dettagli fini.

    alpha, beta: controllano l'intensita' della compressione. alpha e'
    tipicamente una piccola frazione della magnitudine media dei gradienti;
    beta in [0,1] (beta=0 -> nessuna compressione, gradiente invariato).
    """
    grad_mag_avg = np.mean(np.abs(np.diff(source_ch, axis=0))) + 1e-6

    def v(y, x, ny, nx):
        d_s = source_ch[y, x] - source_ch[ny, nx]
        mag = abs(d_s)
        if mag < 1e-6:
            return 0.0
        scale = ((alpha * grad_mag_avg) ** beta) * (mag ** (-beta))
        return scale * d_s
    return v
