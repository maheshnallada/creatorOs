import argparse
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "True"

from pydantic import BaseModel, ConfigDict, Field, model_validator

VARIANTS = ("writer_initial", "editor_final", "cold_start", "learned")
METRIC_NAMES = ("hook_quality", "brand_voice", "relevance")


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1, max_length=100)
    brief: str = Field(min_length=1, max_length=2000)
    brand_voice_profile: dict[str, Any]
    learned_playbook: list[dict[str, Any]] = Field(default_factory=list)
    drafts: dict[str, str]

    @model_validator(mode="after")
    def require_paired_outputs(self):
        if set(self.drafts) != set(VARIANTS):
            raise ValueError(f"drafts must contain exactly these variants: {', '.join(VARIANTS)}")
        if any(not draft.strip() for draft in self.drafts.values()):
            raise ValueError("all benchmark drafts must be nonempty")
        return self


class VariantEvaluation(BaseModel):
    output: str
    metric_scores: dict[str, float]
    composite_score: float


class EvaluationResult(BaseModel):
    case_id: str
    variants: dict[str, VariantEvaluation]
    editor_lift: float
    memory_lift: float


@dataclass
class _OfflineTestCase:
    input: str
    actual_output: str


def _build_geval_metrics(model):
    from deepeval.metrics import GEval
    from deepeval.test_case import SingleTurnParams

    dimensions = (
        (
            "hook_quality",
            "Score whether the opening hook is specific, engaging, and relevant to the brief.",
        ),
        (
            "brand_voice",
            "Score adherence to the supplied brand voice profile and playbook. If no memory is supplied, score whether the output avoids claiming to match unseen voice rules.",
        ),
        (
            "relevance",
            "Score whether the content fulfills the creator brief and target audience needs.",
        ),
    )
    return [
        GEval(
            name=name,
            criteria=criteria,
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
            model=model,
            threshold=0.5,
            async_mode=False,
        )
        for name, criteria in dimensions
    ]


def _evaluation_input(case: EvaluationCase, variant: str) -> str:
    if variant == "cold_start":
        memory = {"profile": None, "playbook": []}
    elif variant == "learned":
        memory = {
            "profile": case.brand_voice_profile,
            "playbook": case.learned_playbook,
        }
    else:
        memory = {"profile": case.brand_voice_profile, "playbook": []}
    return f"Brief:\n{case.brief}\n\nMemory context:\n{json.dumps(memory, ensure_ascii=False)}"


def run_benchmarks(
    cases: list[EvaluationCase],
    model,
    metric_factory: Callable[[Any], list[Any]] | None = None,
) -> list[EvaluationResult]:
    offline_metrics = metric_factory is not None
    make_metrics = metric_factory or _build_geval_metrics
    if offline_metrics:
        test_case_type = _OfflineTestCase
    else:
        from deepeval.test_case import LLMTestCase

        test_case_type = LLMTestCase

    results = []
    for case in cases:
        variant_results = {}
        for variant in VARIANTS:
            metrics = make_metrics(model)
            test_case = test_case_type(
                input=_evaluation_input(case, variant),
                actual_output=case.drafts[variant],
            )
            scores = {}
            for metric in metrics:
                metric.measure(test_case)
                score = float(metric.score)
                scores[metric.name] = max(0.0, min(1.0, score))
            variant_results[variant] = VariantEvaluation(
                output=case.drafts[variant],
                metric_scores=scores,
                composite_score=sum(scores.values()) / len(scores),
            )

        results.append(
            EvaluationResult(
                case_id=case.case_id,
                variants=variant_results,
                editor_lift=(
                    variant_results["editor_final"].composite_score
                    - variant_results["writer_initial"].composite_score
                ),
                memory_lift=(
                    variant_results["learned"].composite_score
                    - variant_results["cold_start"].composite_score
                ),
            )
        )
    return results


def write_results(results: list[EvaluationResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([result.model_dump(mode="json") for result in results], indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score saved CreatorOS draft pairs with Mesh-backed DeepEval"
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="JSON file containing actual paired drafts"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/results/phase3.json"),
        help="Path for measured benchmark results",
    )
    parser.add_argument("--model", default="openai/omni-moderation-latest", help="Mesh model ID for the judge")
    args = parser.parse_args(argv)

    from app.config import Settings
    from app.llm.mesh_client import MeshClient
    from eval.mesh_model import MeshDeepEvalModel

    settings = Settings()
    if settings.mesh_api_key is None:
        parser.error("MESH_API_KEY is required; no benchmark results were written")
    cases = [EvaluationCase.model_validate(item) for item in json.loads(args.input.read_text())]
    mesh_client = MeshClient(settings)
    try:
        model = MeshDeepEvalModel(mesh_client, args.model)
        results = run_benchmarks(cases, model)
        write_results(results, args.output)
    finally:
        mesh_client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
