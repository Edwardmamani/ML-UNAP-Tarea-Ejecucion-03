"""Construye (y opcionalmente ejecuta) los 5 notebooks de la Tarea de Ejecución 03.

Uso:
    .venv/bin/python src/build_notebooks.py            # construye y ejecuta todos
    .venv/bin/python src/build_notebooks.py 1 3        # solo los ítems indicados
    .venv/bin/python src/build_notebooks.py --no-exec  # solo construye
"""
import importlib
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nb_common import new_notebook  # noqa: E402


def build(item: int, execute: bool = True):
    mod = importlib.import_module(f"nb_item{item}")
    nb = new_notebook(mod.cells())
    out = ROOT / "notebooks" / mod.FILENAME
    out.parent.mkdir(exist_ok=True)
    if execute:
        t0 = time.time()
        NotebookClient(nb, timeout=1800, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT)}}).execute()
        print(f"  Ítem {item} ejecutado en {time.time() - t0:.1f}s")
    nbformat.write(nb, out)
    print(f"  -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    items = [int(a) for a in args] or [1, 2, 3, 4, 5]
    for i in items:
        print(f"Construyendo ítem {i}...")
        build(i, execute="--no-exec" not in sys.argv)
