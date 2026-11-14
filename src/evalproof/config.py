"""Suite configuration: one YAML file describes the data, the system, the scorers and the gate."""

from __future__ import annotations

import difflib
import hashlib
import re
from pathlib import Path
from typing import Any, Literal, get_args

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from evalproof.dataset import canonical_json

ScorerType = Literal[
    "exact_match",
    "contains",
    "regex",
    "json_schema",
    "numeric_tolerance",
    "recall_at_k",
    "mrr",
    "ndcg",
    "llm_judge",
    "checks",
    "python",
]


class ConfigError(ValueError):
    pass


class Pricing(BaseModel):
    """USD per million tokens. Local models default to zero."""

    model_config = ConfigDict(extra="forbid")

    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.input_per_mtok + tokens_out * self.output_per_mtok) / 1_000_000


class GenerationParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: float = 0.0
    seed: int | None = 0
    max_tokens: int = 512
    think: bool | None = False  # reasoning models: off by default for speed and determinism
    extra: dict[str, Any] = Field(default_factory=dict)


class TargetConfig(BaseModel):
    """A model or system to call. ``target`` is one of:

    * ``ollama:<model>`` or ``anthropic:<model>``, the two native adapters;
    * ``<profile>:<model>`` for an OpenAI-compatible service: ``openai``, ``azure``,
      ``gemini``, ``mistral``, ``groq``, ``together``, ``fireworks``, ``deepseek``, ``xai``,
      ``openrouter``, ``cerebras``, ``vllm``, ``lmstudio``, ``llamacpp``, or
      ``openai-compatible`` for any other server (it needs ``base_url``). The model name is
      everything after the first colon, so ``openrouter:anthropic/claude-sonnet-4.5`` works;
    * ``python:<module>:<function>``;
    * an ``http(s)://`` endpoint.

    ``api_key_env`` names the environment variable that holds the key, replacing the
    provider's usual one. ``headers`` are extra HTTP headers for every provider call (for
    example an ``anthropic-version`` override or a gateway routing header). Keys belong in
    environment variables: ``headers`` never reaches a run file or the cache (and is not part
    of the cache key), but a suite file is usually committed, so a secret placed in it would be
    committed too.
    """

    model_config = ConfigDict(extra="forbid")

    target: str
    label: str | None = None  # how reports name this system, e.g. "qwen3:1.7b + grounded prompt"
    system_prompt: str | None = None
    prompt_template: str = "{input}"
    params: GenerationParams = Field(default_factory=GenerationParams)
    timeout_s: float = 120.0
    base_url: str | None = None
    api_key_env: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    pricing: Pricing = Field(default_factory=Pricing)

    @property
    def kind(self) -> str:
        if self.target.startswith(("http://", "https://")):
            return "http"
        return self.target.split(":", 1)[0]


class ScorerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: ScorerType
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_params(self) -> ScorerConfig:
        # The scorers with a fixed vocabulary are checked when the suite loads, not mid-run.
        if self.type in ("checks", "python"):
            from evalproof.scorers import validate_scorer_params  # imports this module

            validate_scorer_params(self.type, self.params)
        return self


class CalibrationConfig(BaseModel):
    """How to calibrate a judge: against a labels file, or against another (trusted) scorer
    sampled down to ``gold_sample`` cases to mimic a small human-reviewed set."""

    model_config = ConfigDict(extra="forbid")

    judge_scorer: str
    labels: str | None = None
    gold_scorer: str | None = None
    gold_sample: int | None = None

    @model_validator(mode="after")
    def _one_source(self) -> CalibrationConfig:
        if (self.labels is None) == (self.gold_scorer is None):
            raise ValueError("calibration needs exactly one of 'labels' or 'gold_scorer'")
        return self


class GateConfig(BaseModel):
    """Fail only when a drop is both larger than ``max_drop`` and statistically significant.

    ``metric`` gates one scorer. Add ``metrics`` to gate several; their p-values are then
    Holm-corrected so the chance of a false failure across the set stays at ``alpha``.
    Either may be omitted as long as one scorer is named; ``gated`` is the full list.
    """

    model_config = ConfigDict(extra="forbid")

    metric: str | None = None
    metrics: list[str] = Field(default_factory=list)
    max_drop: float = 0.02
    alpha: float = 0.05
    min_cases: int = 20
    max_p95_latency_ms: float | None = None
    max_cost_usd: float | None = None
    n_boot: int = 5000
    # A changed judge (target, CLI, isolation or resolved model) makes two runs' scores
    # incomparable, so the gate refuses unless told to accept it. A new CLI version alone only
    # warns, so a pinned CLI version is advisory.
    allow_judge_change: bool = False
    # Absolute checks on the candidate alone (``evalproof run --gate`` applies just these).
    # ``min_pass_rate`` is the lowest candidate pass rate per gated metric (a float for all of
    # them, or a dict by scorer name); errored cases count as failures. ``must_pass`` lists
    # case tags: a case with one must pass every gated metric. ``max_new_failures`` is the most
    # cases that may go from pass to fail on a binary gated metric, whatever the p-value; 0
    # suits deterministic scorers.
    min_pass_rate: float | dict[str, float] | None = None
    must_pass: list[str] = Field(default_factory=list)
    max_new_failures: int | None = None

    @model_validator(mode="after")
    def _check_absolute(self) -> GateConfig:
        floors = self.min_pass_rate
        for value in floors.values() if isinstance(floors, dict) else [floors]:
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"min_pass_rate must be between 0 and 1, got {value}")
        if self.max_new_failures is not None and self.max_new_failures < 0:
            raise ValueError("max_new_failures must be >= 0")
        return self

    @model_validator(mode="after")
    def _name_a_metric(self) -> GateConfig:
        names = self.gated
        if not names:
            raise ValueError("a gate needs 'metric' or 'metrics'")
        # Keep ``gate.metric`` set for code that gates a single scorer.
        self.metric, self.metrics = names[0], names[1:]
        return self

    @property
    def gated(self) -> list[str]:
        """The scorers to gate on, ``metric`` first, without repeats."""
        names = [self.metric] if self.metric else []
        return list(dict.fromkeys([*names, *self.metrics]))

    @property
    def pass_rate_floors(self) -> dict[str, float]:
        """The minimum pass rate per scorer: a float applies to every gated metric."""
        if self.min_pass_rate is None:
            return {}
        if isinstance(self.min_pass_rate, dict):
            return dict(self.min_pass_rate)
        return {name: self.min_pass_rate for name in self.gated}

    @property
    def has_absolute_checks(self) -> bool:
        """True when the gate sets a check that needs no baseline."""
        return self.min_pass_rate is not None or bool(self.must_pass)


VARIANT_NAME = re.compile(r"[a-z0-9._-]+")

# The parts of a system override that merge into the suite's own rather than replace it.
_MERGED_SYSTEM_KEYS = ("params", "pricing")


class VariantConfig(BaseModel):
    """One way to run the suite: the same cases, scorers and gate, with another model, prompt or
    flag. ``vars`` name what differs (they reach a python or http system through ``RunContext``
    and the request body, and a model prompt as template fields); ``system`` is an optional
    partial ``TargetConfig`` merged over the suite's ``system``."""

    model_config = ConfigDict(extra="forbid")

    name: str
    vars: dict[str, Any] = Field(default_factory=dict)
    system: dict[str, Any] | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not VARIANT_NAME.fullmatch(v):
            raise ValueError(f"variant name {v!r} must use only a-z, 0-9, '.', '_' and '-'")
        return v

    @field_validator("vars")
    @classmethod
    def _json_vars(cls, v: dict[str, Any]) -> dict[str, Any]:
        try:
            canonical_json(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"vars must be plain JSON values: {exc}") from exc
        return v


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """``override`` over ``base``; a dict meeting a dict is merged key by key, anything else is
    replaced."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def merge_system(base: TargetConfig, override: dict[str, Any]) -> TargetConfig:
    """The suite's ``system`` with a variant's partial override applied.

    ``params`` and ``pricing`` merge key by key (a variant that sets only ``temperature`` keeps the
    suite's ``max_tokens``); every other key, ``headers`` included, is replaced by the override's.
    Raises ``ValidationError`` when the result is not a valid target."""
    merged = base.model_dump()
    for key, value in override.items():
        if key in _MERGED_SYSTEM_KEYS and isinstance(value, dict):
            merged[key] = _merge(merged.get(key) or {}, value)
        else:
            merged[key] = value
    return TargetConfig.model_validate(merged)


class SuiteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    dataset: str
    system: TargetConfig
    scorers: list[ScorerConfig]
    judge: TargetConfig | None = None
    calibration: CalibrationConfig | None = None
    gate: GateConfig | None = None
    variants: list[VariantConfig] = Field(default_factory=list)
    seed: int = 0
    cache_dir: str = ".evalproof/cache"
    concurrency: int = 1
    base_dir: str = "."

    @model_validator(mode="after")
    def _check_variants(self) -> SuiteConfig:
        names = [v.name for v in self.variants]
        if len(set(names)) != len(names):
            raise ValueError(f"variant names must be unique, got {names}")
        for variant in self.variants:
            if variant.system is None:
                continue
            try:
                merge_system(self.system, variant.system)
            except ValidationError as exc:
                problems = format_validation_error(exc, f"variant {variant.name!r}", TargetConfig)
                raise ValueError(problems.replace("\n", "; ")) from exc
        return self

    @model_validator(mode="after")
    def _check(self) -> SuiteConfig:
        names = [s.name for s in self.scorers]
        if len(set(names)) != len(names):
            raise ValueError(f"scorer names must be unique, got {names}")
        if any(s.type == "llm_judge" for s in self.scorers) and self.judge is None:
            raise ValueError("an llm_judge scorer needs a 'judge' target")
        if self.gate:
            for i, name in enumerate(self.gate.gated):
                if name not in names:
                    where = "gate.metric" if i == 0 else "gate.metrics"
                    raise ValueError(f"{where} {name!r} is not a scorer name")
            if isinstance(self.gate.min_pass_rate, dict):
                for name in self.gate.min_pass_rate:
                    if name not in names:
                        raise ValueError(f"gate.min_pass_rate {name!r} is not a scorer name")
        if self.calibration:
            for field in ("judge_scorer", "gold_scorer"):
                ref = getattr(self.calibration, field)
                if ref is not None and ref not in names:
                    raise ValueError(f"calibration.{field} {ref!r} is not a scorer name")
        if self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        return self

    def resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else Path(self.base_dir) / p

    def variant(self, name: str) -> VariantConfig:
        """The variant called ``name``; a ``ConfigError`` listing the names when there is none."""
        for v in self.variants:
            if v.name == name:
                return v
        known = ", ".join(v.name for v in self.variants) or "none defined"
        raise ConfigError(f"unknown variant {name!r}; the suite's variants: {known}")

    def variant_system(self, name: str | None) -> TargetConfig:
        """The system a variant runs: the suite's with the variant's override merged over it.
        ``None`` is the suite as written."""
        if name is None:
            return self.system
        override = self.variant(name).system
        return self.system if override is None else merge_system(self.system, override)


def _target_identity(cfg: TargetConfig | None) -> dict[str, Any] | None:
    """What decides a target's answers. ``label``, ``timeout_s``, ``headers``, ``api_key_env``
    and ``pricing`` change how it is reported or reached, not what it says."""
    if cfg is None:
        return None
    return {
        "target": cfg.target,
        "system_prompt": cfg.system_prompt,
        "prompt_template": cfg.prompt_template,
        "params": cfg.params.model_dump(mode="json"),
        "base_url": cfg.base_url,
    }


def suite_fingerprint(
    cfg: SuiteConfig, system_cfg: TargetConfig | None = None, variant: str | None = None
) -> str:
    """A short hash of what makes a run's scores: the system under test, the judge, every
    scorer and the seed. ``system_cfg`` is the target that actually ran, overrides included;
    it defaults to the suite's own, with the variant's override merged in when ``variant`` names
    one. A variant adds its name and vars to the hash; a run without one hashes exactly as it did
    before variants existed.

    The dataset is not in it (``evalproof verify`` checks it case by case), nor is anything that
    only names or paces the run: the labels, timeouts, headers, key variable, pricing, gate,
    cache directory, concurrency and suite name.
    """
    material: dict[str, Any] = {
        "system": _target_identity(system_cfg or cfg.variant_system(variant)),
        "judge": _target_identity(cfg.judge),
        "scorers": [s.model_dump(mode="json") for s in cfg.scorers],
        "seed": cfg.seed,
    }
    if variant is not None:
        material["variant"] = {"name": variant, "vars": cfg.variant(variant).vars}
    return hashlib.sha256(canonical_json(material).encode()).hexdigest()[:16]


def _model_in(annotation: Any) -> type[BaseModel] | None:
    """The pydantic model inside an annotation such as ``GateConfig | None`` or ``list[X]``."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for arg in get_args(annotation):
        found = _model_in(arg)
        if found is not None:
            return found
    return None


def format_validation_error(
    exc: ValidationError, path: str | Path, model: type[BaseModel] | None = None
) -> str:
    """One readable line per problem, ``<path>: <where>: <what>``, with no pydantic URLs. An
    unknown key gets a suggestion when one is close, and the valid keys otherwise."""
    lines = []
    for err in exc.errors(include_url=False):
        loc = err["loc"]
        msg = err["msg"].removeprefix("Value error, ")
        if err["type"] == "extra_forbidden":
            current: type[BaseModel] | None = model or SuiteConfig
            for part in loc[:-1]:
                if isinstance(part, str) and current is not None:
                    field = current.model_fields.get(part)
                    current = _model_in(field.annotation) if field else None
            valid = list(current.model_fields) if current else []
            if current is TargetConfig:  # system and judge also take prompt text from a file
                valid += ["system_prompt_file", "prompt_template_file"]
            bad = str(loc[-1])
            close = difflib.get_close_matches(bad, valid, n=1)
            if close:
                msg = f"unknown key (did you mean {close[0]!r}?)"
            else:
                msg = f"unknown key (valid keys: {', '.join(valid)})" if valid else "unknown key"
        where = ".".join(str(part) for part in loc)
        lines.append(f"{path}: {where}: {msg}" if where else f"{path}: {msg}")
    return "\n".join(lines)


def load_config(path: str | Path) -> SuiteConfig:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"suite file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    raw.setdefault("base_dir", str(path.parent))
    # Prompt and rubric text may live in files next to the suite.
    blocks = [raw.get("system"), raw.get("judge")]
    variants = raw.get("variants")
    if isinstance(variants, list):  # a variant's system override takes prompt files too
        blocks += [v.get("system") for v in variants if isinstance(v, dict)]
    for block in blocks:
        if isinstance(block, dict):
            for key in ("system_prompt", "prompt_template"):
                file_key = f"{key}_file"
                if file_key in block:
                    block[key] = (path.parent / block.pop(file_key)).read_text(encoding="utf-8")
    try:
        return SuiteConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(format_validation_error(exc, path)) from exc
