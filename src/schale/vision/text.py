"""CPU ONNX CTC recognition shared by student and inventory screen profiles."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class TextReading:
    text: str
    score: float
    character_scores: tuple[float, ...] = ()


def prepare(patch):
    from PIL import Image

    if patch is None or patch.ndim != 3 or patch.shape[2] != 3 or not patch.size:
        raise ValueError("Text input must be a nonempty BGR image")
    h, w = patch.shape[:2]
    ratio = min(4, int(w / h) + 1)
    size = ((64, 64), (96, 48), (112, 40), (128, 32))[ratio - 1]
    rgb = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
    array = np.asarray(rgb.resize(size, Image.Resampling.BICUBIC)).copy()
    return array.transpose(2, 0, 1).astype(np.float32) / 127.5 - 1


class CtcRecognizer:
    def __init__(self, model: Path, charset: list[str]):
        import onnxruntime as ort

        self.charset = charset
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            str(model), options, providers=["CPUExecutionProvider"]
        )

    def infer(self, inputs) -> list[TextReading]:
        probabilities = self.session.run(None, {"image": inputs})[0]
        if not isinstance(probabilities, np.ndarray):
            raise RuntimeError("Text model returned a non-tensor output")
        if probabilities.ndim != 3 or probabilities.shape[0] != len(inputs):
            raise RuntimeError("Text model returned an invalid batch shape")
        if (
            probabilities.shape[2] != len(self.charset)
            or not np.isfinite(probabilities).all()
        ):
            raise RuntimeError("Text model returned invalid CTC probabilities")
        readings = []
        for prob in probabilities:
            ids = prob.argmax(1)
            selected = (ids != 0) & np.r_[True, ids[1:] != ids[:-1]]
            scores = prob[np.arange(len(ids)), ids][selected]
            readings.append(
                TextReading(
                    "".join(self.charset[i] for i in ids[selected]),
                    float(scores.min()) if len(scores) else 0.0,
                    tuple(float(score) for score in scores),
                )
            )
        return readings

    def read(self, patches: list) -> list[TextReading]:
        groups: dict[tuple, list] = {}
        for index, patch in enumerate(patches):
            array = prepare(patch)
            groups.setdefault(array.shape, []).append((index, array))
        readings = [TextReading("", 0.0) for _ in patches]
        for group in groups.values():
            for (index, _), reading in zip(
                group, self.infer(np.stack([array for _, array in group])), strict=True
            ):
                readings[index] = reading
        return readings
