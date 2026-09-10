"""Unit tests for the v0.4.10 original-Qwen-repo runtime repairs in app.asr.

The v0.4.8 field report ("recording made, ASR does nothing with it") was root
caused as THREE stacked incompatibilities between the ORIGINAL Qwen/Qwen3-ASR
repo layout and transformers >= 5.0:

1. ``preprocessor_config.json`` declares a Whisper feature extractor that
   never pads the mel time axis for Qwen3ASREncoder (the 938-vs-100 error);
2. ``config.json`` nests everything under ``thinker_config`` so AutoConfig
   falls back to DEFAULT sub-config shapes (text 2048 vs 1024, audio 1024
   vs 896);
3. the checkpoint keys carry the ``thinker.`` prefix so plain
   from_pretrained matches ZERO weights (silently random model).

``_parse_thinker_layout`` detects layout 2 (pure stdlib, json only),
``_repaired_qwen_config`` builds the true config, ``_ORIGINAL_QWEN_KEY_MAPPING``
repairs layout 3 (transformers WeightRenaming search+replace semantics,
specific rules first), ``_ensure_qwen3_feature_extractor`` repairs layout 1
and ``_resolve_n_window`` feeds it the encoder's real n_window.

Pure standard library + injected fake transformers modules — no
torch/transformers needed (the imports in app.asr are guarded and degrade
gracefully in the sandbox). Recreated verbatim after the sandbox restore
events dropped the untracked working-tree file (see the worklog incident
notes: restores keep tracked files but drop fresh untracked ones).
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, Optional


def _thinker_payload() -> dict[str, Any]:
    """The original Qwen/Qwen3-ASR-0.6B config.json shape (abridged)."""
    return {
        "thinker_config": {
            "text_config": {
                "_name_or_path": "Qwen/Qwen3-ASR-0.6B",
                "architectures": ["Qwen3ASRForConditionalGeneration"],
                "hidden_size": 1024,
                "num_hidden_layers": 18,
            },
            "audio_config": {
                "_name_or_path": "Qwen/Qwen3-ASR-0.6B",
                "model_type": "qwen3_asr_audio_encoder",
                "d_model": 896,
                "encoder_layers": 18,
                "n_window": 50,
            },
            "audio_token_id": 151647,
            "audio_start_token_id": 151646,
            "audio_end_token_id": 151647,
        }
    }


class ParseThinkerLayoutTests(unittest.TestCase):
    """_parse_thinker_layout: original repo -> cleaned sub-configs, else None."""

    def _layout(
        self, payload: Optional[Any] = None, raw_text: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        from app.asr import _parse_thinker_layout

        with tempfile.TemporaryDirectory() as tmp:
            if raw_text is not None:
                Path(tmp, "config.json").write_text(raw_text, encoding="utf-8")
            elif payload is not None:
                Path(tmp, "config.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            return _parse_thinker_layout(tmp)

    def test_original_layout_cleaned(self) -> None:
        """thinker_config parsed; noise keys dropped; model types normalised."""
        layout = self._layout(_thinker_payload())
        self.assertIsNotNone(layout)
        assert layout is not None  # for the type checker
        self.assertEqual(layout["text"]["hidden_size"], 1024)
        self.assertEqual(layout["text"]["num_hidden_layers"], 18)
        self.assertEqual(layout["audio"]["d_model"], 896)
        self.assertEqual(layout["audio"]["encoder_layers"], 18)
        # cleaned for Qwen3ASRConfig consumption
        self.assertNotIn("_name_or_path", layout["text"])
        self.assertNotIn("architectures", layout["text"])
        self.assertNotIn("_name_or_path", layout["audio"])
        self.assertNotIn("architectures", layout["audio"])
        # token contract carried over
        self.assertEqual(
            layout["tokens"],
            {
                "audio_token_id": 151647,
                "audio_start_token_id": 151646,
                "audio_end_token_id": 151647,
            },
        )

    def test_model_type_normalised(self) -> None:
        """Text keeps an existing model_type, defaults to qwen3; the encoder
        model_type is FORCIBLY normalised to the registered class name."""
        payload = _thinker_payload()
        payload["thinker_config"]["text_config"]["model_type"] = "qwen3_lm"
        layout = self._layout(payload)
        self.assertIsNotNone(layout)
        assert layout is not None
        # setdefault semantics: an existing text model_type is preserved
        self.assertEqual(layout["text"]["model_type"], "qwen3_lm")
        # the original repo says qwen3_asr_audio_encoder; transformers
        # registers the encoder class as qwen3_asr_encoder
        self.assertEqual(layout["audio"]["model_type"], "qwen3_asr_encoder")
        # missing text model_type gets the same default
        payload = _thinker_payload()
        payload["thinker_config"]["text_config"].pop("model_type", None)
        layout = self._layout(payload)
        assert layout is not None
        self.assertEqual(layout["text"]["model_type"], "qwen3")

    def test_hf_layout_returns_none(self) -> None:
        """The -hf repo layout (top-level sub-configs) loads natively."""
        payload = {
            "text_config": {"hidden_size": 1024},
            "audio_config": {"d_model": 896, "n_window": 50},
        }
        self.assertIsNone(self._layout(payload))

    def test_hub_id_or_missing_config_returns_none(self) -> None:
        """A hub id / empty dir has no config.json — nothing to repair."""
        # empty temporary directory
        self.assertIsNone(self._layout())
        # a bare hub id path that does not exist at all
        from app.asr import _parse_thinker_layout

        self.assertIsNone(_parse_thinker_layout("Qwen/Qwen3-ASR-0.6B"))
        self.assertIsNone(_parse_thinker_layout("no/such/dir"))

    def test_corrupt_json_returns_none(self) -> None:
        self.assertIsNone(self._layout(raw_text="{ not valid json"))

    def test_non_dict_thinker_returns_none(self) -> None:
        self.assertIsNone(self._layout({"thinker_config": "thinker"}))
        self.assertIsNone(self._layout({"thinker_config": ["list"]}))

    def test_sparse_tokens(self) -> None:
        """Only the present audio-token ids land in the tokens dict."""
        payload = _thinker_payload()
        del payload["thinker_config"]["audio_start_token_id"]
        del payload["thinker_config"]["audio_end_token_id"]
        layout = self._layout(payload)
        self.assertIsNotNone(layout)
        assert layout is not None
        self.assertEqual(layout["tokens"], {"audio_token_id": 151647})
        # none present -> empty tokens, layout still parsed
        payload = _thinker_payload()
        for key in (
            "audio_token_id",
            "audio_start_token_id",
            "audio_end_token_id",
        ):
            del payload["thinker_config"][key]
        layout = self._layout(payload)
        assert layout is not None
        self.assertEqual(layout["tokens"], {})


class OriginalRepoKeyMappingTests(unittest.TestCase):
    """_ORIGINAL_QWEN_KEY_MAPPING: WeightRenaming search+replace semantics."""

    def _mapping(self) -> "dict[str, str]":
        from app.asr import _ORIGINAL_QWEN_KEY_MAPPING

        return dict(_ORIGINAL_QWEN_KEY_MAPPING)

    def _apply(self, key: str) -> str:
        """transformers renames: first matching rule (in order) wins."""
        import re

        for pattern, replacement in self._mapping().items():
            if re.search(pattern, key):
                return re.sub(pattern, replacement, key)
        return key

    def test_all_five_rules(self) -> None:
        """Every original-repo key class lands on its transformers target."""
        cases = {
            "thinker.audio_tower.proj1.weight": (
                "model.multi_modal_projector.linear_1.weight"
            ),
            "thinker.audio_tower.proj2.weight": (
                "model.multi_modal_projector.linear_2.weight"
            ),
            "thinker.audio_tower.blocks.0.attn.q_proj.weight": (
                "model.audio_tower.blocks.0.attn.q_proj.weight"
            ),
            "thinker.model.layers.0.mlp.gate_proj.weight": (
                "model.language_model.layers.0.mlp.gate_proj.weight"
            ),
            "thinker.lm_head.weight": "lm_head.weight",
        }
        for raw, expected in cases.items():
            self.assertEqual(self._apply(raw), expected, msg=raw)

    def test_specific_rules_precede_generic(self) -> None:
        """proj1/proj2 must be remapped BEFORE the generic audio_tower rule.

        Checked both on the RAW mapping order (insertion order of the dict)
        and through an applied key: if the generic rule ran first, proj
        weights would land inside the audio tower and match nothing.
        """
        rules = list(self._mapping().items())
        self.assertEqual(rules[0][0], r"^thinker\.audio_tower\.proj1\.")
        self.assertEqual(rules[1][0], r"^thinker\.audio_tower\.proj2\.")
        self.assertEqual(rules[2][0], r"^thinker\.audio_tower\.")
        # applied: the projector path wins over model.audio_tower.proj1...
        self.assertEqual(
            self._apply("thinker.audio_tower.proj1.bias"),
            "model.multi_modal_projector.linear_1.bias",
        )
        self.assertEqual(
            self._apply("thinker.audio_tower.proj2.bias"),
            "model.multi_modal_projector.linear_2.bias",
        )

    def test_unmatched_keys_unchanged(self) -> None:
        """transformers-layout and unrelated keys pass through verbatim."""
        for key in (
            "model.language_model.embed_tokens.weight",
            "model.audio_tower.blocks.3.norm.weight",
            "lm_head.weight",
            "thinker_wrapper.foo",
        ):
            self.assertEqual(self._apply(key), key, msg=key)

    def test_patterns_anchored_at_start(self) -> None:
        """^ anchoring: a mid-string 'thinker.' segment must NOT match."""
        self.assertEqual(
            self._apply("prefix.thinker.model.layers.0"), "prefix.thinker.model.layers.0"
        )
        self.assertEqual(
            self._apply("x.thinker.audio_tower.proj1.weight"),
            "x.thinker.audio_tower.proj1.weight",
        )

    def test_suffix_preserved(self) -> None:
        """The whole tail after the matched prefix survives the rename."""
        self.assertEqual(
            self._apply(
                "thinker.model.layers.17.self_attn.q_proj.bias"
            ),
            "model.language_model.layers.17.self_attn.q_proj.bias",
        )
        self.assertEqual(
            self._apply("thinker.audio_tower.blocks.11.mlp.up_proj.weight"),
            "model.audio_tower.blocks.11.mlp.up_proj.weight",
        )


class ResolveNWindowTests(unittest.TestCase):
    """_resolve_n_window: the encoder sub-config value, default 50."""

    def test_audio_config_object(self) -> None:
        """The -hf layout: config.audio_config (attribute object)."""
        from app.asr import _resolve_n_window

        model = types.SimpleNamespace(
            config=types.SimpleNamespace(
                audio_config=types.SimpleNamespace(n_window=42)
            )
        )
        self.assertEqual(_resolve_n_window(model), 42)

    def test_thinker_dict_of_dicts(self) -> None:
        """The original repo: thinker_config as dict-of-dicts pre-load form."""
        from app.asr import _resolve_n_window

        model = types.SimpleNamespace(
            config=types.SimpleNamespace(
                thinker_config={"audio_config": {"n_window": 33}}
            )
        )
        self.assertEqual(_resolve_n_window(model), 33)

    def test_config_first_order(self) -> None:
        """A value on the config node itself outranks the sub-configs."""
        from app.asr import _resolve_n_window

        model = types.SimpleNamespace(
            config=types.SimpleNamespace(
                n_window=11,
                audio_config=types.SimpleNamespace(n_window=22),
            )
        )
        self.assertEqual(_resolve_n_window(model), 11)

    def test_invalid_values_fall_back_to_default(self) -> None:
        """bool / negative / string / missing are rejected; default 50."""
        from app.asr import _resolve_n_window

        for bad in (True, -5, "50", None):
            model = types.SimpleNamespace(
                config=types.SimpleNamespace(
                    audio_config=types.SimpleNamespace(n_window=bad)
                )
            )
            self.assertEqual(_resolve_n_window(model), 50, msg=repr(bad))
        # no config object at all
        self.assertEqual(_resolve_n_window(object()), 50)


class _FakeQwen3FeatureExtractor:
    """Records constructor kwargs; stands in for the transformers class."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class EnsureFeatureExtractorTests(unittest.TestCase):
    """_ensure_qwen3_feature_extractor: the Whisper->Qwen3 swap."""

    def setUp(self) -> None:
        self._saved = sys.modules.get("transformers")

    def tearDown(self) -> None:
        if self._saved is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = self._saved

    def _inject(self, fe_cls: Any = _FakeQwen3FeatureExtractor) -> None:
        mod = types.ModuleType("transformers")
        mod.Qwen3ASRFeatureExtractor = fe_cls  # type: ignore[attr-defined]
        sys.modules["transformers"] = mod

    def _whisper_fe(self, **overrides: Any) -> Any:
        params = {
            "feature_size": 80,
            "sampling_rate": 16000,
            "hop_length": 160,
            "n_fft": 400,
            "chunk_length": 30,
            "padding_value": 0.0,
            "dither": 0.0,
            "return_attention_mask": True,
        }
        params.update(overrides)
        return types.SimpleNamespace(**params)

    def _model(self, n_window: Any = 42) -> Any:
        return types.SimpleNamespace(
            config=types.SimpleNamespace(
                audio_config=types.SimpleNamespace(n_window=n_window)
            )
        )

    def test_swap_carries_params_and_min_length(self) -> None:
        """Whisper params carried over; n_window from the model; 8000 min."""
        from app.asr import _ensure_qwen3_feature_extractor

        self._inject()
        processor = types.SimpleNamespace(feature_extractor=self._whisper_fe())
        _ensure_qwen3_feature_extractor(processor, self._model(42))
        fe = processor.feature_extractor
        self.assertIsInstance(fe, _FakeQwen3FeatureExtractor)
        assert isinstance(fe, _FakeQwen3FeatureExtractor)
        self.assertEqual(
            fe.kwargs,
            {
                "feature_size": 80,
                "sampling_rate": 16000,
                "hop_length": 160,
                "n_fft": 400,
                "chunk_length": 30,
                "padding_value": 0.0,
                "dither": 0.0,
                "return_attention_mask": True,
                "n_window": 42,
                "min_length": 8000,  # original Qwen3-ASR library default
            },
        )

    def test_idempotent_noop_when_already_qwen(self) -> None:
        """An already-correct extractor is left alone (identity kept)."""
        from app.asr import _ensure_qwen3_feature_extractor

        self._inject()
        existing = _FakeQwen3FeatureExtractor(feature_size=128)
        processor = types.SimpleNamespace(feature_extractor=existing)
        _ensure_qwen3_feature_extractor(processor, self._model())
        self.assertIs(processor.feature_extractor, existing)

    def test_garbage_current_falls_back_to_defaults(self) -> None:
        """A current extractor with no usable attrs -> sane defaults."""
        from app.asr import _ensure_qwen3_feature_extractor

        self._inject()
        processor = types.SimpleNamespace(feature_extractor=types.SimpleNamespace())
        _ensure_qwen3_feature_extractor(processor, self._model())
        fe = processor.feature_extractor
        self.assertIsInstance(fe, _FakeQwen3FeatureExtractor)
        assert isinstance(fe, _FakeQwen3FeatureExtractor)
        self.assertEqual(fe.kwargs["feature_size"], 128)
        self.assertEqual(fe.kwargs["sampling_rate"], 16000)
        self.assertEqual(fe.kwargs["hop_length"], 160)
        self.assertEqual(fe.kwargs["n_fft"], 400)
        self.assertEqual(fe.kwargs["n_window"], 42)
        # missing current -> early return, no swap at all
        empty = types.SimpleNamespace()  # type: ignore[attr-defined]
        _ensure_qwen3_feature_extractor(empty, self._model())
        self.assertFalse(hasattr(empty, "feature_extractor"))

    def test_both_slots_replaced(self) -> None:
        """audio_processor AND feature_extractor are both swapped."""
        from app.asr import _ensure_qwen3_feature_extractor

        self._inject()
        whisper = self._whisper_fe()
        processor = types.SimpleNamespace(
            audio_processor=whisper, feature_extractor=whisper
        )
        _ensure_qwen3_feature_extractor(processor, self._model())
        self.assertIsInstance(processor.audio_processor, _FakeQwen3FeatureExtractor)
        self.assertIsInstance(processor.feature_extractor, _FakeQwen3FeatureExtractor)
        self.assertIsNot(processor.audio_processor, whisper)
        self.assertIsNot(processor.feature_extractor, whisper)

    def test_exploding_ctor_never_raises(self) -> None:
        """A repair failure is swallowed — the load must never die."""

        class _Exploding:
            def __init__(self, **kwargs: Any) -> None:
                raise RuntimeError("boom")

        self._inject(_Exploding)
        processor = types.SimpleNamespace(feature_extractor=self._whisper_fe())
        try:
            from app.asr import _ensure_qwen3_feature_extractor

            _ensure_qwen3_feature_extractor(processor, self._model())
        except Exception as exc:  # noqa: BLE001 — the whole point of the test
            self.fail(f"_ensure_qwen3_feature_extractor raised: {exc!r}")
        # the original (Whisper) extractor stays in place
        self.assertIs(
            processor.feature_extractor.__dict__.get("feature_size"), 80
        )


class RepairedQwenConfigTests(unittest.TestCase):
    """_repaired_qwen_config: true shapes from the repo's own thinker values."""

    def setUp(self) -> None:
        self._saved = sys.modules.get("transformers")

    def tearDown(self) -> None:
        if self._saved is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = self._saved

    def test_builds_true_config_from_thinker_values(self) -> None:
        from app.asr import _repaired_qwen_config

        class _FakeQwen3Config:
            def __init__(
                self,
                audio_config: Any = None,
                text_config: Any = None,
                **tokens: Any,
            ) -> None:
                self.audio_config = types.SimpleNamespace(
                    **{"d_model": 0, "encoder_layers": 0, "n_window": 0, **(audio_config or {})}
                )
                self.text_config = types.SimpleNamespace(
                    **{"hidden_size": 0, "num_hidden_layers": 0, **(text_config or {})}
                )
                self.tokens = tokens

        mod = types.ModuleType("transformers")
        mod.Qwen3ASRConfig = _FakeQwen3Config  # type: ignore[attr-defined]
        sys.modules["transformers"] = mod

        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text(
                json.dumps(_thinker_payload()), encoding="utf-8"
            )
            config = _repaired_qwen_config(tmp)
        self.assertIsNotNone(config)
        assert config is not None
        # the REAL values from the repo (not the AutoConfig defaults
        # 2048/1024) — this is what makes the weight remap line up
        self.assertEqual(config.text_config.hidden_size, 1024)
        self.assertEqual(config.text_config.num_hidden_layers, 18)
        self.assertEqual(config.audio_config.d_model, 896)
        self.assertEqual(config.audio_config.encoder_layers, 18)
        self.assertEqual(config.audio_config.n_window, 50)
        self.assertEqual(config.tokens.get("audio_token_id"), 151647)

        # the -hf layout loads natively — no repair object, plain path
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text(
                json.dumps({"text_config": {}, "audio_config": {}}),
                encoding="utf-8",
            )
            self.assertIsNone(_repaired_qwen_config(tmp))


if __name__ == "__main__":
    unittest.main()
