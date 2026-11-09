"""Strictly local, pinned CED-Mini audio tagging; broad tags are not SFX decisions."""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np

from dotears.audiosep import file_digest
from dotears.errors import ConfigurationError, DependencyError, MediaError

CED_REVISION = "26c3ebcae85d4330f4fc26763f029539a3afcda0"
CED_FILES = {
    "model.onnx": "2a86374dbe1fa03e96b0acf6a02618ac71852d12aa3d72a030b062ff1e5c5434",
    "config.json": "4036cba16f3759f5588e731029327969ff30a475fe23268fa698530cd7fd10ea",
}


class LocalCEDOnnx:
    """CPU inference without remote Python/model loaders or network access."""

    def __init__(self, folder: Path, threads: int = 4):
        if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 16:
            raise ConfigurationError("CED threads must be between 1 and 16.")
        self.folder = folder.resolve()
        for name, expected in CED_FILES.items():
            path = (self.folder / name).resolve()
            if (
                not path.is_relative_to(self.folder)
                or not path.is_file()
                or file_digest(path) != expected
            ):
                raise ConfigurationError(f"Unprepared or modified CED artifact: {name}")
        config = json.loads((self.folder / "config.json").read_text("utf-8"))
        self.labels = [config["id2label"][str(index)] for index in range(527)]
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise DependencyError("CED requires the optional CPU onnxruntime package.") from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
        self.session = ort.InferenceSession(
            str(self.folder / "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        if [item.name for item in self.session.get_inputs()] != ["waveform"] or len(
            self.session.get_outputs()
        ) != 1:
            raise ConfigurationError("Unexpected CED ONNX interface.")

    def predict(self, waveform: np.ndarray, rate: int = 16000) -> np.ndarray:
        waveform = np.asarray(waveform, dtype=np.float32)
        if (
            waveform.ndim != 1
            or rate != 16000
            or not 2048 <= len(waveform) <= 160000
            or not np.isfinite(waveform).all()
        ):
            raise ConfigurationError("CED needs finite 16kHz mono audio, 0.128–10 seconds.")
        values = self.session.run(None, {"waveform": waveform[None]})[0][0]
        if (
            values.shape != (527,)
            or not np.isfinite(values).all()
            or np.min(values) < -1e-5
            or np.max(values) > 1.00001
        ):
            raise MediaError("CED returned invalid broad-tag scores.")
        return np.clip(values, 0, 1)


FLOAT_FILES = {
    "configuration_ced.py": "147ac3602c1b35c3309d68575e3f19eb22cc2afe088897397991613e68a82a64",
    "modeling_ced.py": "c8a8e8b865e4e3dcebffefb40215d6778bfa380ecea388f11806e33521a29327",
    "feature_extraction_ced.py": "d2eff0a8cbc563196f8ccef4180d9e9299d643bfee087a545e97e9c4a9e64cb4",
    "model.safetensors": "e4070d02ca53ec7a1df83bbf779a0e70bc2bbc0a9bacde6330e1dc37c990b215",
    "config.json": CED_FILES["config.json"],
    "preprocessor_config.json": "668fbf4f00f9e3915de04fc07a621434ef1746b933662840623a9701291ca55a",
}


class LocalCED:
    """Float32 inspected upstream forward; the quantized export failed local parity."""

    def __init__(self, folder: Path, threads: int = 4):
        if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 16:
            raise ConfigurationError("CED threads must be between 1 and 16.")
        folder = folder.resolve()
        for name, expected in FLOAT_FILES.items():
            path = (folder / name).resolve()
            if (
                not path.is_relative_to(folder)
                or not path.is_file()
                or file_digest(path) != expected
            ):
                raise ConfigurationError(f"Unprepared or modified CED float artifact: {name}")
        try:
            import torch
            from safetensors.torch import load_file

            importlib.import_module("torchaudio.transforms")
        except ImportError as exc:
            raise DependencyError(
                "Float CED requires CPU torch, torchaudio and transformers."
            ) from exc
        torch.set_num_threads(threads)
        self.torch = torch
        package_name = "_dotears_ced_mini_26c3ebc"
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = [str(folder)]
            sys.modules[package_name] = package
            for name in ("configuration_ced", "modeling_ced", "feature_extraction_ced"):
                module = types.ModuleType(package_name + "." + name)
                module.__file__, module.__package__ = str(folder / (name + ".py")), package_name
                sys.modules[module.__name__] = module
                # Only the hash-inspected Apache-2.0 implementation, no remote loaders.
                exec(
                    compile(Path(module.__file__).read_text("utf-8"), module.__file__, "exec"),
                    module.__dict__,
                )
        config_module = sys.modules[package_name + ".configuration_ced"]
        data = json.loads((folder / "config.json").read_text("utf-8"))
        self.labels = [data["id2label"][str(i)] for i in range(527)]
        # The upstream constructor otherwise downloads an unrelated label CSV. Load
        # the already pinned label map locally before constructing any neural layers.
        config = config_module.CedConfig(**{**data, "outputdim": 1})
        config.outputdim = 527
        config.id2label = {i: value for i, value in enumerate(self.labels)}
        config.label2id = {value: i for i, value in enumerate(self.labels)}
        cls = sys.modules[package_name + ".modeling_ced"].CedForAudioClassification
        self.model = cls(config).cpu().eval()
        self.model.load_state_dict(load_file(str(folder / "model.safetensors")), strict=True)
        extractor = sys.modules[package_name + ".feature_extraction_ced"].CedFeatureExtractor
        self.extractor = extractor(**json.loads((folder / "preprocessor_config.json").read_text()))

    def predict(self, waveform: np.ndarray, rate: int = 16000) -> np.ndarray:
        waveform = np.asarray(waveform, dtype=np.float32)
        if (
            waveform.ndim != 1
            or rate != 16000
            or not 2048 <= len(waveform) <= 160000
            or not np.isfinite(waveform).all()
        ):
            raise ConfigurationError("CED needs finite 16kHz mono audio, 0.128–10 seconds.")
        inputs = self.extractor(waveform, sampling_rate=rate, return_tensors="pt")
        with self.torch.inference_mode():
            values = self.model(**inputs).logits[0].cpu().numpy()
        if (
            values.shape != (527,)
            or not np.isfinite(values).all()
            or np.min(values) < -1e-5
            or np.max(values) > 1.00001
        ):
            raise MediaError("CED returned invalid broad-tag scores.")
        return np.clip(values, 0, 1)

    def embedding(self, waveform: np.ndarray, rate: int = 16000) -> np.ndarray:
        """Normalized 256-dimensional raw-audio encoding before the broad-tag head."""
        waveform = np.asarray(waveform, dtype=np.float32)
        if (
            waveform.ndim != 1
            or rate != 16000
            or not 2048 <= len(waveform) <= 160000
            or not np.isfinite(waveform).all()
        ):
            raise ConfigurationError("CED needs finite 16kHz mono audio, 0.128–10 seconds.")
        inputs = self.extractor(waveform, sampling_rate=rate, return_tensors="pt")
        with self.torch.inference_mode():
            values = self.model.encoder(**inputs).logits.mean(dim=1)
            values = self.torch.nn.functional.normalize(values, dim=-1)[0].cpu().numpy()
        if values.shape != (256,) or not np.isfinite(values).all():
            raise MediaError("Invalid CED latent representation.")
        return values
