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
from poisson import poisson_blend

DATA_DIR = Path("data")

# Registro dei metodi: nome sottocomando -> funzione di blending.
# Per aggiungere un nuovo metodo in futuro basta implementare la funzione
# (stessa firma: source, target, mask -> risultato) e aggiungerla qui.
METHODS = {
    "cloning": poisson_blend,
    # "mixed": mixed_gradients_blend,
    # "flatten": texture_flattening_blend,
    # "illumination": illumination_change_blend,
}


def resolve_paths(args, method_name: str):
    """Determina i path di source/mask/target/output.

    Se --test è specificato, usa data/testN/{source,mask,target}.png e come
    output data/testN/result_<method>.png, a meno che --output non sia dato esplicitamente.
    Altrimenti richiede --source/--mask/--target espliciti.
    """
    if args.test is not None:
        test_dir = DATA_DIR / f"test{args.test}"
        
        # Funzione di supporto per cercare il file con estensioni diverse
        def find_file(base_name, default_ext=".png"):
            for ext in [".png", ".jpg", ".jpeg"]:
                p = test_dir / f"{base_name}{ext}"
                if p.exists():
                    return p
            return test_dir / f"{base_name}{default_ext}" # Fallback se nessuno esiste

        source = args.source or find_file("source")
        mask = args.mask or find_file("mask")
        target = args.target or find_file("target")
        output = args.output or test_dir / f"result_{method_name}.png"

    else:
        missing = [name for name in ("source", "mask", "target") if getattr(args, name) is None]
        if missing:
            raise SystemExit(
                f"Devi specificare --test N oppure tutti i path espliciti "
                f"(mancano: {', '.join('--' + m for m in missing)})"
            )
        source, mask, target = args.source, args.mask, args.target
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

    return parser.parse_args()


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

    print(f"[4/4] Eseguo metodo '{args.method}'...")
    t0 = time.time()
    #esegue la funzione di blending corrispondente al metodo scelto
    result = blend_fn(source, target, mask)            
    print(f"      Fatto in {time.time() - t0:.2f}s")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(result, output_path)
    print(f"Risultato salvato in: {output_path}")


if __name__ == "__main__":
    main()