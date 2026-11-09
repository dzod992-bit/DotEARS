"""Pinned CPU FUSS graph, used as a query-free feature view, not an SFX recognizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dotears.audiosep import file_digest
from dotears.errors import ConfigurationError, DependencyError, MediaError

FILES = {
    "baseline_dry_inference.meta": (
        "5081a27aafda15e38a11fb27c9d6670aec52fe969b61823b77db80690da52821"
    ),
    "baseline_dry_model.data-00000-of-00001": (
        "e8c15c81fef9fd37195051274e6f2954f66134ce5523b7d591cc1626175d1fb2"
    ),
    "baseline_dry_model.index": "748a8c3b12a2e7e52acfc19463c557f972e1f64a6da263892ab15dc03ce77217",
}


class LocalFUSS:
    def __init__(self, folder: Path, threads=4):
        if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 16:
            raise ConfigurationError("FUSS CPU threads must be between 1 and 16.")
        folder = folder.resolve()
        if (folder / "baseline_dry_model").is_dir():
            folder = folder / "baseline_dry_model"
        for name, expected in FILES.items():
            path = (folder / name).resolve()
            if (
                not path.is_relative_to(folder)
                or not path.is_file()
                or file_digest(path) != expected
            ):
                raise ConfigurationError("Unprepared or modified FUSS artifact: " + name)
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise DependencyError(
                "FUSS feature views require the prepared local TensorFlow environment."
            ) from exc
        self.graph = tf.Graph()
        definition = tf.compat.v1.MetaGraphDef()
        definition.ParseFromString((folder / "baseline_dry_inference.meta").read_bytes())
        if any(
            n.op in ("PyFunc", "PyFuncStateless", "EagerPyFunc") for n in definition.graph_def.node
        ):
            raise ConfigurationError("Custom Python graph operations are not supported.")
        config = tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=threads,
            inter_op_parallelism_threads=1,
            device_count={"GPU": 0},
        )
        self.session = tf.compat.v1.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            saver = tf.compat.v1.train.import_meta_graph(
                str(folder / "baseline_dry_inference.meta"), clear_devices=True
            )
            saver.restore(self.session, str(folder / "baseline_dry_model"))
        self.input = self.graph.get_tensor_by_name("input_audio/receiver_audio:0")
        self.output = self.graph.get_tensor_by_name("denoised_waveforms:0")

    def separate(self, waveform: np.ndarray) -> np.ndarray:
        values = np.asarray(waveform, dtype=np.float32)
        if values.ndim != 1 or not 2048 <= len(values) <= 160000 or not np.isfinite(values).all():
            raise ConfigurationError("FUSS needs finite 16kHz mono audio, 0.128–10 seconds.")
        result = self.session.run(self.output, {self.input: values.reshape(1, 1, -1)})[0]
        if result.shape != (4, len(values)) or not np.isfinite(result).all():
            raise MediaError("Invalid FUSS component view.")
        if np.max(np.abs(result.sum(0) - values)) > 1e-3:
            raise MediaError("FUSS component views do not reconstruct the input mixture.")
        return result

    def close(self):
        self.session.close()
