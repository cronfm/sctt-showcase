"""Execute the reviewed notebook in a fresh kernel and check its final assertions."""

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager
import sys
from tempfile import TemporaryDirectory
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="Optional executed notebook; the source is never overwritten by default"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / "notebooks/train_sctt.ipynb", as_version=4)
    nbformat.validate(notebook)
    # Use this interpreter even if a global 'python3' kernel points elsewhere.
    with TemporaryDirectory(prefix="sctt-kernel-") as directory:
        kernel_dir = Path(directory) / "sctt-demo"
        kernel_dir.mkdir()
        (kernel_dir / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "SCTT demo",
                    "language": "python",
                }
            ),
            encoding="utf-8",
        )
        manager = KernelSpecManager(kernel_dirs=[directory])
        from jupyter_client import KernelManager

        kernel = KernelManager(kernel_name="sctt-demo", kernel_spec_manager=manager)
        client = NotebookClient(notebook, timeout=180, km=kernel, resources={"metadata": {"path": str(root)}})
        try:
            client.execute()
        finally:
            if kernel.has_kernel:
                kernel.shutdown_kernel(now=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, args.output)
    print(f"Executed {sum(c.cell_type == 'code' for c in notebook.cells)} notebook code cells successfully.")
    for cell in notebook.cells:
        if "verification" in cell.get("metadata", {}).get("tags", []):
            for output in cell.get("outputs", []):
                if output.output_type == "stream":
                    print(output.text.strip())


if __name__ == "__main__":
    main()
