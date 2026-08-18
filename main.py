"""
CLI per Poisson Image Editing, con sottocomandi per ciascun metodo implementato.

Uso rapido (dataset pronto in data/testN/{source,mask,target}.png):
    python main.py cloning --test 1

Uso con path custom (per immagini tue, fuori da data/):
    python main.py cloning --source path/source.png --mask path/mask.png \
                            --target path/target.png --output path/result.png

Metodi disponibili (per ora): cloning
Metodi futuri (stesso pattern): mixed, flatten, illumination, tiling, feathering
"""
import argparse
import time
from pathlib import Path

from io_utils import load_image, load_mask, save_image
from variants import (
    poisson_blend,
    mixed_gradients_blend,
    texture_flattening_blend,
    illumination_change_blend,
)

DATA_DIR = Path("data")

# Registro dei metodi: nome sottocomando -> funzione di blending.
# Per aggiungere un nuovo metodo in futuro basta implementare la funzione in
# variants.py (stessa firma: source, target, mask, **kwargs -> risultato) e
# aggiungerla qui. Se ha parametri extra, aggiungerli anche a EXTRA_ARGS sotto.
METHODS = {
    "cloning": poisson_blend,
    "mixed": mixed_gradients_blend,
    "flatten": texture_flattening_blend,
    "illumination": illumination_change_blend,
    # "tiling": seamless_tiling_blend,  # diverso dagli altri: preprocessing dei bordi, non un guidance field
}

# Parametri CLI specifici per metodo, oltre a quelli comuni (--test, --source, ...).
# Formato: nome_metodo -> lista di dict passati direttamente ad add_argument.
EXTRA_ARGS = {
    "flatten": [
        {"flags": ["--sigma"], "type": float, "default": 1.5,
         "help": "Sigma del filtro gaussiano per l'edge detector Canny (default 1.5)"},
        {"flags": ["--flatten-factor"], "type": float, "default": 3.0,
         "help": "Fattore di flattening per l'appiattimento della texture (default 3.0)"},
    ],
    "illumination": [
        {"flags": ["--alpha"], "type": float, "default": 0.2,
         "help": "Parametro alpha di compressione del gradiente (default 0.2)"},
        {"flags": ["--beta"], "type": float, "default": 0.2,
         "help": "Parametro beta di compressione del gradiente, in [0,1] (default 0.2)"},
    ],
}

# Metodi che lavorano IN-PLACE su un'unica immagine + maschera (nessun compositing
# tra due immagini diverse): texture flattening e illumination change nel paper si
# applicano a source stessa (source == target). Per questi non serve un file
# "source" separato: se manca, si usa direttamente target.
SELF_EDIT_METHODS = {"flatten", "illumination"}

def _find_file(test_dir: Path, base_name: str, required: bool = True):
    """Cerca test_dir/base_name.{png,jpg,jpeg}. Se required=False e non trova
    nulla, ritorna None invece di sollevare un errore."""
    for ext in (".png", ".jpg", ".jpeg"):
        p = test_dir / f"{base_name}{ext}"
        if p.exists():
            return p
    if required:
        raise SystemExit(f"Non trovo '{base_name}' (.png/.jpg/.jpeg) in {test_dir}")
    return None


def resolve_paths(args, method_name: str):
    """Determina i path di source/mask/target/output.

    Se --test è specificato, usa data/testN/{source,mask,target}.png/jpg/jpeg
    e come output data/testN/result_<method>.png, a meno che --output non sia
    dato esplicitamente. Altrimenti richiede --mask/--target espliciti
    (--source è opzionale per i metodi in SELF_EDIT_METHODS).
    """
    is_self_edit = method_name in SELF_EDIT_METHODS

    if args.test is not None:
        test_dir = DATA_DIR / ("illum1" if args.test == 6 else f"test{args.test}")

        mask = args.mask or _find_file(test_dir, "mask")
        target = args.target or _find_file(test_dir, "target")

        if args.source:
            source = args.source
        else:
            # per i metodi self-edit la source è opzionale: se manca, usa target
            found = _find_file(test_dir, "source", required=not is_self_edit)
            source = found or target

        output = args.output or test_dir / f"result_{method_name}.png"
    else:
        required_names = ["mask", "target"] if is_self_edit else ["source", "mask", "target"]
        missing = [name for name in required_names if getattr(args, name) is None]
        if missing:
            raise SystemExit(
                f"Devi specificare --test N oppure {'--mask/--target' if is_self_edit else 'tutti i path espliciti'} "
                f"(mancano: {', '.join('--' + m for m in missing)})"
            )
        mask, target = args.mask, args.target
        source = args.source or (target if is_self_edit else None)
        output = args.output or Path("result.png")

    return Path(source), Path(mask), Path(target), Path(output)

def add_common_args(subparser):
    """Argomenti condivisi da tutti i sottocomandi (metodi)."""
    subparser.add_argument(
        "--test", type=int, default=None, metavar="N",
        help="Usa il dataset pronto data/testN/ (es. --test 1 -> data/test1/)",
    )
    subparser.add_argument("--source", type=Path, default=None, help="Override path sorgente")
    subparser.add_argument("--mask", type=Path, default=None, help="Override path maschera")
    subparser.add_argument("--target", type=Path, default=None, help="Override path destinazione")
    subparser.add_argument("--output", type=Path, default=None, help="Override path output")


def parse_args():
    parser = argparse.ArgumentParser(description="Poisson Image Editing")
    subparsers = parser.add_subparsers(dest="method", required=True, help="Metodo da eseguire")

    for name in METHODS:
        sp = subparsers.add_parser(name, help=f"Esegui il metodo '{name}'")
        add_common_args(sp)
        for extra in EXTRA_ARGS.get(name, []):
            flags = extra["flags"]
            kwargs = {k: v for k, v in extra.items() if k != "flags"}
            sp.add_argument(*flags, **kwargs)

    return parser.parse_args()


def extra_kwargs(args, method_name: str) -> dict:
    """Estrae dagli args solo i parametri specifici del metodo (EXTRA_ARGS),
    da passare come **kwargs alla funzione di blending."""
    kwargs = {}
    for extra in EXTRA_ARGS.get(method_name, []):
        # nome del parametro = flag senza i due trattini iniziali, es. "--sigma" -> "sigma"
        dest = extra["flags"][0].lstrip("-").replace("-", "_")
        kwargs[dest] = getattr(args, dest)
    return kwargs


def main():
    args = parse_args()
    blend_fn = METHODS[args.method]

    source_path, mask_path, target_path, output_path = resolve_paths(args, args.method)

    print(f"[1/4] Carico source: {source_path}")
    source = load_image(source_path)
    print(f"[2/4] Carico target: {target_path}")
    target = load_image(target_path)
    print(f"[3/4] Carico mask:   {mask_path}")
    mask = load_mask(mask_path)
    print(f"      Pixel in Omega: {mask.sum()}")

    kwargs = extra_kwargs(args, args.method)
    print(f"[4/4] Eseguo metodo '{args.method}'{f' {kwargs}' if kwargs else ''}...")
    t0 = time.time()
    result = blend_fn(source, target, mask, **kwargs)
    print(f"      Fatto in {time.time() - t0:.2f}s")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(result, output_path)
    print(f"Risultato salvato in: {output_path}")


if __name__ == "__main__":
    main()