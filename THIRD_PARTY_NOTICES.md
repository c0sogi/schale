# Third-party materials

The MIT license in `LICENSE` applies to the original schale source code. It does
not relicense third-party models, game artwork, trademarks, or downloaded data.

## Blue Archive and SchaleDB

The Python package contains no game artwork, UI excerpts, derived reference
descriptors, learned model weights, user recordings, or account exports. Runtime
skill icons, portraits, and game metadata are fetched from https://schaledb.com/
and cached locally when the corresponding tool is used. Student extraction uses
geometric UI registration without game captures. Optional local vision resources
are supplied separately by the user; installation does not grant rights to
use or redistribute their contents. This project is not an official Blue Archive,
NEXON, or SchaleDB product.

Source of game materials: Blue Archive, NEXON / NEXON Games.
See https://bluearchive.nexon.com/ and NEXON's Game IP guide:
https://member.nexon.com/policy/policywrapper.aspx?policytype=37 .
No permission to redistribute game materials is represented by this project.

## Optional equipment model

The inventory reader supports a locally installed MobileNetV3-Small ONNX model
and its class mapping. Neither weights nor training images are distributed in
the source or wheel. The training code can initialize from TorchVision's
ImageNet weights. TorchVision's source license does not establish the rights to
every pretrained weight or its training data; review the terms of the assets you
choose to use or distribute:
https://github.com/pytorch/vision/blob/main/docs/source/models.rst .

## Student numeric model

The student numeric ONNX model is downloaded automatically from this project's
`numeric-runtime-v1` GitHub release, not embedded in the Python wheel or source
distribution. The download is pinned by SHA-256 and size in the source. It was exported from OpenOCR
(https://github.com/Topdu/OpenOCR), revision
`0d522801ec6dc1df852c6b6d4ed6a08f5127ed97`.
The bundle retains `LICENSE-OpenOCR` (Apache-2.0) and `NOTICE.md`, plus source and
checkpoint hashes in `metadata.json`. The ONNX conversion and confidence-threshold
calibration were performed for this extraction tool. Use trusted model bundles;
checking a hash against adjacent metadata detects corruption, not publisher identity.

## Runtime dependencies

OpenCV, ONNX Runtime, NumPy, Pillow, Requests, Pydantic, Typer and their dependencies
are installed separately and retain their respective licenses. Student and inventory
inference do not require PyTorch or EasyOCR. PyTorch is confined to the optional
source training group. Student video decoding uses system FFmpeg/FFprobe when
available, or the separately installed PyAV dependency and its FFmpeg libraries.
PyAV and its binary wheels retain their respective third-party notices/licenses.
