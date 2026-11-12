"""Suite configuration: one YAML file describes the data, the system, the scorers and the gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

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
]


class ConfigError(ValueError):
    pass


class Pricing(BaseModel):
    """USD per million tokens. Local models default to zero."""

    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.input_per_mtok + tokens_out * self.output_per_mtok) / 1_000_000


class GenerationParams(BaseModel):
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
    name: str
    type: ScorerType
    params: dict[str, Any] = Field(default_factory=dict)


class CalibrationConfig(BaseModel):
    """How to calibrate a judge: against a labels file, or against another (trusted) scorer
    sampled down to ``gold_sample`` cases to mimic a small human-reviewed set."""

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
    """Fail only when a drop is both larger than ``max_drop`` and statistically significant."""

    metric: str
    max_drop: float = 0.02
    alpha: float = 0.05
    min_cases: int = 20
    max_p95_latency_ms: float | None = None
    max_cost_usd: float | None = None
    n_boot: int = 5000


class SuiteConfig(BaseModel):
    name: str
    dataset: str
    system: TargetConfig
    scorers: list[ScorerConfig]
    judge: TargetConfig | None = None
    calibration: CalibrationConfig | None = None
    gate: GateConfig | None = None
    seed: int = 0
    cache_dir: str = ".evalproof/cache"
    concurrency: int = 1
    base_dir: str = "."

    @model_validator(mode="after")
    def _check(self) -> SuiteConfig:
        names = [s.name for s in self.scorers]
        if len(set(names)) != len(names):
            raise ValueError(f"scorer names must be unique, got {names}")
        if any(s.type == "llm_judge" for s in self.scorers) and self.judge is None:
            raise ValueError("an llm_judge scorer needs a 'judge' target")
        if self.gate and self.gate.metric not in names:
            raise ValueError(f"gate.metric {self.gate.metric!r} is not a scorer name")
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
    for section in ("system", "judge"):
        block = raw.get(section)
        if isinstance(block, dict):
            for key in ("system_prompt", "prompt_template"):
                file_key = f"{key}_file"
                if file_key in block:
                    block[key] = (path.parent / block.pop(file_key)).read_text(encoding="utf-8")
    try:
        return SuiteConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
