import os
import shutil
from pathlib import Path
import argparse

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("GENDER_BIAS_ROOT", DEFAULT_ROOT))

def collect_plots(overwrite: bool = False, symlink: bool = False, model: str | None = None):
    """
    Copie (ou symlink) tous les .png depuis results/** vers results/plots/** 
    en conservant la structure relative.
    """
    results_dir = ROOT / "results"
    plots_root = results_dir / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)

    # Parcours de tous les fichiers .png sous results/
    for png in results_dir.rglob("*.png"):
        # on ignore déjà la destination 'results/plots'
        try:
            png.relative_to(plots_root)
            # si on arrive ici, c'est déjà sous plots -> on saute
            continue
        except ValueError:
            pass

        # Filtrer par modèle si demandé (match sur le chemin)
        if model and model not in str(png):
            continue

        # Destination qui préserve la structure : results/plots/<chemin_relatif_depuis_results>
        rel = png.relative_to(results_dir)
        dst = plots_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists():
            if not overwrite:
                continue
            dst.unlink()

        if symlink:
            try:
                dst.symlink_to(png)
            except FileExistsError:
                pass
        else:
            shutil.copy2(png, dst)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="Écraser les fichiers déjà existants dans plots/")
    parser.add_argument("--symlink", action="store_true", help="Créer des liens symboliques au lieu de copier")
    parser.add_argument("--model", type=str, default=None, help="Filtrer uniquement les PNG dont le chemin contient ce modèle")
    args = parser.parse_args()

    collect_plots(overwrite=args.overwrite, symlink=args.symlink, model=args.model)
    print("C'ePlots collectés dans: ", ROOT / "results" / "plots")

if __name__ == "__main__":
    main()
