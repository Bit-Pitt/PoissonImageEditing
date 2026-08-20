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
    seamless_tiling_blend,
    texture_flattening_blend,
    illumination_change_blend,
    local_color_change_blend,
    border_feathering_blend,
)

from tile_preprocessing import make_2x2_tile_preview

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
    "color": local_color_change_blend,
    "tiling": seamless_tiling_blend,
    "feather": border_feathering_blend,
    
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
    "color": [
    {"flags": ["--hue-shift"], "type": float, "default": 0.0,
     "help": "Spostamento della tonalita' in [0,1], ciclico (default 0.0)"},
    ],
    "feather": [
        {"flags": ["--alpha"], "type": float, "default": 0.5,
        "help": "Peso del blending sul bordo: 0=nessun feathering, 1=bordo tutto source (default 0.5)"},
        {"flags": ["--feather-width"], "type": int, "default": 3,
        "help": "Spessore in pixel dell'anello di bordo sfumato (default 3)"},
    ],
}

# Metodi che lavorano IN-PLACE su un'unica immagine + maschera (nessun compositing
# tra due immagini diverse): texture flattening e illumination change nel paper si
# applicano a source stessa (source == target). Per questi non serve un file
# "source" separato: se manca, si usa direttamente target.
SELF_EDIT_METHODS = {"flatten", "illumination","color","tiling"}

# Metodi che non hanno bisogno di una mask.png da file: la maschera viene
# generata internamente dalla funzione (es. tiling: Omega = tutta l'immagine
# tranne l'anello esterno).
NO_MASK_METHODS = {"tiling"}

# Override del nome file da cercare, quando diverso da "target"/"source"/"mask".
FILE_BASENAME_OVERRIDES = {
    "tiling": {"target": "texture"},
}

# Prefisso di cartella diverso da "test", per metodo.
TEST_DIR_PREFIXES = {"tiling": "texture"}

# Mapping esplicito test-number -> nome cartella, per i metodi con directory
# "speciali" non numerate in sequenza (illum1, color1).
TEST_DIR_OVERRIDES = {6: "illum1", 7: "color1"}

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

# Per accorpare i test "speciali" (illum1, color1) con i test numerati, restituisce la cartella
def _resolve_test_dir(args, method_name: str) -> Path:
    if method_name in TEST_DIR_PREFIXES:
        return DATA_DIR / f"{TEST_DIR_PREFIXES[method_name]}{args.test}"
    if args.test in TEST_DIR_OVERRIDES:
        return DATA_DIR / TEST_DIR_OVERRIDES[args.test]
    return DATA_DIR / f"test{args.test}"


def resolve_paths(args, method_name: str):
    """Determina i path di source/mask/target/output."""
    is_self_edit = method_name in SELF_EDIT_METHODS
    needs_mask = method_name not in NO_MASK_METHODS
    target_basename = FILE_BASENAME_OVERRIDES.get(method_name, {}).get("target", "target")

    if args.test is not None:
        test_dir = _resolve_test_dir(args, method_name)

        mask = (args.mask or _find_file(test_dir, "mask")) if needs_mask else None
        target = args.target or _find_file(test_dir, target_basename)

        if args.source:
            source = args.source
        else:
            found = _find_file(test_dir, "source", required=not is_self_edit)
            source = found or target

        output = args.output or test_dir / f"result_{method_name}.png"
    else:
        required_names = ["target"] if is_self_edit else ["source", "target"]
        if needs_mask:
            required_names.append("mask")
        missing = [name for name in required_names if getattr(args, name) is None]
        if missing:
            raise SystemExit(
                f"Devi specificare --test N oppure i path richiesti "
                f"(mancano: {', '.join('--' + m for m in missing)})"
            )
        mask = args.mask if needs_mask else None
        target = args.target
        source = args.source or (target if is_self_edit else None)
        output = args.output or Path("result.png")

    return (
        Path(source) if source else None,
        Path(mask) if mask else None,
        Path(target) if target else None,
        Path(output),
    )

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
    
    if mask_path is not None:
        print(f"[3/4] Carico mask:   {mask_path}")
        mask = load_mask(mask_path)
        print(f"      Pixel in Omega: {mask.sum()}")
    else:
        print("[3/4] Nessuna mask da file: generata internamente dal metodo")
        mask = None

    kwargs = extra_kwargs(args, args.method)
    print(f"[4/4] Eseguo metodo '{args.method}'{f' {kwargs}' if kwargs else ''}...")
    t0 = time.time()
    result = blend_fn(source, target, mask, **kwargs)
    print(f"      Eseguito in {time.time() - t0:.2f}s")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(result, output_path)
    print(f"Risultato salvato in: {output_path}")


    if args.method == "tiling":
        preview = make_2x2_tile_preview(result)
        preview_path = output_path.parent / "tiling_2x2_preview.png"
        save_image(preview, preview_path)
        print(f"Anteprima 2x2 salvata in: {preview_path}")


if __name__ == "__main__":
    main()