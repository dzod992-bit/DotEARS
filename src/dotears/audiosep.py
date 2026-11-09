"""Optional, strictly local AudioSep research adapter for one inspected checkpoint.

The pinned MIT-licensed separator implementation stays in the explicitly prepared model
directory. Only its two hash-checked modules are loaded, under private module names.
No training framework, unused audio encoder or remote model code is imported.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

from dotears.errors import ConfigurationError, DependencyError, MediaError

SOURCE_REVISION = "944583f18b84589dc965de3ad77525c945334252"
PINNED_FILES = {
    "audiosep_base_4M_steps.ckpt": (
        "f8cda01bfd0ebd141eef45d41db7a3ada23a56568465840d3cff04b8010ce82c"
    ),
    "source/models/base.py": "a27aacd64271411f5fe08b109da721be50314340ce93aa0d12d4556339dbb896",
    "source/models/resunet.py": "5583de4d980b2a0e7622d8accfcef3480151341186b44595a7787fcb48611a9b",
    "source/models/CLAP/open_clip/model.py": (
        "a875be52eefee5909e6c338a9e321fd4003f97c80af56871522a6e65a9664b68"
    ),
}


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024**2):
            value.update(chunk)
    return value.hexdigest()


def verify_prepared_audiosep(folder: Path) -> dict:
    folder = folder.resolve()
    preparation = json.loads((folder / "preparation.json").read_text("utf-8"))
    if preparation["github_revision"] != SOURCE_REVISION:
        raise ConfigurationError("This adapter supports only the inspected AudioSep revision.")
    for relative, expected in PINNED_FILES.items():
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder) or not path.is_file() or file_digest(path) != expected:
            raise ConfigurationError(f"Unprepared or modified AudioSep artifact: {relative}")
    tokenizer = json.loads((folder / "tokenizer/manifest.json").read_text("utf-8"))
    for row in tokenizer["files"]:
        path = (folder / "tokenizer" / row["path"]).resolve()
        if not path.is_relative_to(folder / "tokenizer") or file_digest(path) != row["sha256"]:
            raise ConfigurationError("Prepared RoBERTa configuration/tokenizer changed.")
    return preparation


def _upstream_separator(folder: Path):
    base_name = "_dotears_audiosep_base_944583f"
    resunet_name = "_dotears_audiosep_resunet_944583f"
    base_path = folder / "source/models/base.py"
    if base_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(base_name, base_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[base_name] = module
        spec.loader.exec_module(module)
    if resunet_name not in sys.modules:
        path = folder / "source/models/resunet.py"
        source = path.read_text("utf-8")
        original = "from models.base import Base, init_layer, init_bn, act"
        if source.count(original) != 1:
            raise ConfigurationError("Unexpected upstream separator imports.")
        # Relocate exactly one absolute import; model code and parameter names are unchanged.
        source = source.replace(original, f"from {base_name} import Base, init_layer, init_bn, act")
        spec = importlib.util.spec_from_loader(resunet_name, loader=None)
        module = importlib.util.module_from_spec(spec)
        module.__file__ = str(path)
        sys.modules[resunet_name] = module
        exec(compile(source, str(path), "exec"), module.__dict__)
    return sys.modules[resunet_name].ResUNet30


class LocalAudioSep:
    """CPU inference; the caller must keep original mixed audio and evidence provenance."""

    def __init__(self, folder: Path, threads: int = 4):
        if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 16:
            raise ConfigurationError("AudioSep CPU threads must be between 1 and 16.")
        self.folder = folder.resolve()
        self.preparation = verify_prepared_audiosep(self.folder)
        try:
            import torch
            from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
        except ImportError as exc:
            raise DependencyError(
                "AudioSep needs its separate CPU torch/transformers environment."
            ) from exc
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        self.torch = torch
        started = time.perf_counter()
        checkpoint_path = self.folder / "audiosep_base_4M_steps.ckpt"
        if torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint_path):
            raise ConfigurationError(
                "The inspected separator checkpoint contains unexpected globals."
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
        state = checkpoint["state_dict"]
        separator_state = {
            k.removeprefix("ss_model."): v for k, v in state.items() if k.startswith("ss_model.")
        }
        text_prefix = "query_encoder.model.text_branch."
        text_state = {
            k.removeprefix(text_prefix): v for k, v in state.items() if k.startswith(text_prefix)
        }
        projection_prefix = "query_encoder.model.text_projection."
        projection_state = {
            k.removeprefix(projection_prefix): v
            for k, v in state.items()
            if k.startswith(projection_prefix)
        }
        if (
            not separator_state
            or not text_state
            or set(projection_state) != {"0.weight", "0.bias", "2.weight", "2.bias"}
        ):
            raise ConfigurationError(
                "Missing paired separator/text weights in the AudioSep checkpoint."
            )
        separator_type = _upstream_separator(self.folder)
        self.separator = separator_type(input_channels=1, output_channels=1, condition_size=512)
        self.separator.load_state_dict(separator_state, strict=True, assign=True)
        self.separator.requires_grad_(False).eval()
        config = RobertaConfig.from_json_file(str(self.folder / "tokenizer/config.json"))
        config._attn_implementation = "eager"
        if (config.hidden_size, config.vocab_size, config.max_position_embeddings) != (
            768,
            50265,
            514,
        ):
            raise ConfigurationError("Unexpected RoBERTa-base architecture.")
        self.text_model = RobertaModel(config, add_pooling_layer=True)
        removed_constants = []
        for name in ("embeddings.position_ids", "embeddings.token_type_ids"):
            if name in text_state:
                expected = getattr(self.text_model.embeddings, name.split(".")[1])
                if not torch.equal(text_state[name], expected):
                    raise ConfigurationError("Unexpected stored RoBERTa constant buffer.")
                del text_state[name]
                removed_constants.append(name)
        self.text_model.load_state_dict(text_state, strict=True, assign=True)
        self.text_model.requires_grad_(False).eval()
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(768, 512), torch.nn.ReLU(), torch.nn.Linear(512, 512)
        )
        self.projection.load_state_dict(projection_state, strict=True, assign=True)
        self.projection.requires_grad_(False).eval()
        self.tokenizer = RobertaTokenizer.from_pretrained(
            str(self.folder / "tokenizer"), local_files_only=True
        )
        self.prompt_cache = {}
        self.load_evidence = {
            "checkpoint_sha256": PINNED_FILES["audiosep_base_4M_steps.ckpt"],
            "separator_keys_loaded_strictly": len(separator_state),
            "text_keys_loaded_strictly": len(text_state),
            "projection_keys_loaded_strictly": len(projection_state),
            "validated_constant_buffers_omitted": removed_constants,
            "weights_only": True,
            "mmap": True,
            "paired_encoder_from_same_checkpoint": True,
            "learned_weights_from_existing_unfused_clap_used": False,
            "loading_seconds": round(time.perf_counter() - started, 3),
        }
        del checkpoint, state

    def encode_text(self, prompts: list[str]):
        if (
            not prompts
            or len(prompts) > 16
            or any(not isinstance(p, str) or not p.strip() or len(p) > 1000 for p in prompts)
        ):
            raise ConfigurationError(
                "AudioSep needs 1–16 non-empty text queries of at most 1000 characters."
            )
        torch = self.torch
        singleton = len(prompts) == 1
        tokens = self.tokenizer(
            prompts * 2 if singleton else prompts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.inference_mode():
            pooled = self.text_model(
                input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]
            ).pooler_output
            features = torch.nn.functional.normalize(self.projection(pooled), dim=-1)
            return features[:1] if singleton else features

    def verify_text_forward(self, prompts: list[str]) -> dict:
        """Compare with the inspected upstream CLAP text-forward functions, same weights."""
        torch = self.torch
        path = self.folder / "source/models/CLAP/open_clip/model.py"
        recorded = next(
            r
            for r in self.preparation["files"]
            if Path(r["path"]).as_posix() == "source/models/CLAP/open_clip/model.py"
        )
        if file_digest(path) != recorded["sha256"]:
            raise ConfigurationError("Upstream CLAP reference code changed.")
        tree = ast.parse(path.read_text("utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CLAP")
        methods = [
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name in {"encode_text", "get_text_embedding"}
        ]
        if len(methods) != 2:
            raise ConfigurationError("Unexpected upstream CLAP text functions.")
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        namespace = {"torch": torch, "F": torch.nn.functional}
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[future, *methods], type_ignores=[])),
                str(path),
                "exec",
            ),
            namespace,
        )
        reference_type = type(
            "UpstreamClapTextReference",
            (torch.nn.Module,),
            {name: namespace[name] for name in ("encode_text", "get_text_embedding")},
        )
        reference = reference_type()
        reference.text_branch_type = "roberta"
        reference.text_branch = self.text_model
        reference.text_projection = self.projection
        reference.eval()
        tokens = self.tokenizer(
            prompts * 2 if len(prompts) == 1 else prompts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.inference_mode():
            expected = reference.get_text_embedding(tokens)[: len(prompts)]
            actual = self.encode_text(prompts)
        maximum = float(torch.max(torch.abs(actual - expected)))
        if not torch.allclose(actual, expected, atol=1e-6, rtol=1e-6):
            raise MediaError("AudioSep text adapter differs from upstream text-forward math.")
        return {
            "passed": True,
            "maximum_absolute_error": maximum,
            "scope": "upstream text forward; same strict-loaded checkpoint weights and tokenizer",
        }

    def separate(self, waveform: np.ndarray, prompt: str, sample_rate: int = 32000) -> np.ndarray:
        waveform = np.asarray(waveform, dtype=np.float32)
        if (
            waveform.ndim != 1
            or not np.isfinite(waveform).all()
            or sample_rate != 32000
            or not 4096 <= len(waveform) <= 320000
        ):
            raise ConfigurationError(
                "AudioSep requires finite 32 kHz mono audio, 0.128–10 seconds."
            )
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 1000:
            raise ConfigurationError("Missing or oversized AudioSep text query.")
        if prompt not in self.prompt_cache:
            self.prompt_cache[prompt] = self.encode_text([prompt])
        torch = self.torch
        with torch.inference_mode():
            result = self.separator(
                {
                    "mixture": torch.from_numpy(waveform)[None, None],
                    "condition": self.prompt_cache[prompt],
                }
            )["waveform"]
        output = result[0, 0].cpu().numpy().copy()
        if output.shape != waveform.shape or not np.isfinite(output).all():
            raise MediaError("AudioSep returned a non-finite or misaligned waveform.")
        return output
