"""Cross-platform cold-download/inference smoke test; no private game resources."""

import importlib.util
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np

from schale.students.download import ensure_numeric_model
from schale.students.numeric import NumericReader


def main():
    with tempfile.TemporaryDirectory(prefix="schale-public-smoke-") as temporary:
        os.environ["SCHALE_CACHE_DIR"] = temporary
        os.environ["SCHALE_OFFLINE"] = "0"
        assert not list(Path(temporary).iterdir())
        directory = ensure_numeric_model()
        model = NumericReader(directory)
        patch = np.full((50, 160, 3), 255, np.uint8)
        cv2.putText(patch, "Lv.90", (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        readings = model.read([patch])
        assert len(readings) == 1 and readings[0].text.startswith("Lv.90")
        assert 0 <= readings[0].score <= 1
        assert importlib.util.find_spec("torch") is None
        os.environ["SCHALE_OFFLINE"] = "1"
        assert ensure_numeric_model() == directory
        assert not (Path(temporary) / "vision-resources").exists()
        print(
            "PASS: public cold download, pinned integrity, CPU ONNX inference, offline cache reuse; no Torch/private references"
        )


if __name__ == "__main__":
    main()
