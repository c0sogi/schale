# Student numeric model provenance

Model: SVTRv2-S / RCTC, from OpenOCR (https://github.com/Topdu/OpenOCR).
Source revision: `0d522801ec6dc1df852c6b6d4ed6a08f5127ed97`.
The supplied `LICENSE-OpenOCR` is the upstream Apache License, Version 2.0.

Changes for schale: exported the source checkpoint to ONNX for CPU inference,
verified numerical parity for the four supported input shapes, and calibrated
the acceptance threshold on Blue Archive UI crops. The trained weights were not
retrained for individual students. The charset, threshold, source checkpoint hash,
ONNX hash, and parity measurements are recorded in `metadata.json`.

This model is distributed separately from schale's MIT-licensed source code.
It is used to recognize short UI labels, not to identify students or determine
the position of the UI. Image localization, numeric grammar/range checking,
template agreement, and multi-frame consistency are implemented by schale.

Verified ONNX SHA-256:
`0a3a4a555aaa4830208601dd085b3a266cd6d52f99b66703dd068cf889eb7d04`

This model is not guaranteed to recognize arbitrary recordings perfectly.
Use the review/inspector output for unresolved or conflicting observations.
