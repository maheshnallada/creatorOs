import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_benchmark_modules():
    try:
        from eval.benchmark import (
            EvaluationCase,
            run_benchmarks,
            write_results,
        )
    except ModuleNotFoundError as error:
        pytest.fail(f"evaluation benchmark implementation is missing: {error.name}")
    return EvaluationCase, run_benchmarks, write_results


def load_mesh_model():
    pytest.importorskip("deepeval", reason="install the optional eval extra")
    try:
        from eval.mesh_model import MeshDeepEvalModel
    except ModuleNotFoundError as error:
        pytest.fail(f"Mesh DeepEval adapter implementation is missing: {error.name}")
    return MeshDeepEvalModel


class FakeMeshClient:
    def __init__(self):
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return SimpleNamespace(content="judge response")


def test_mesh_deepeval_model_routes_sync_and_async_calls_through_mesh():
    model_type = load_mesh_model()
    mesh_client = FakeMeshClient()
    model = model_type(mesh_client, model_name="openai/gpt-oss-120b")

    sync_result = model.generate("Score this draft.", schema={"type": "object"})
    async_result = asyncio.run(model.a_generate("Score this revision."))

    assert sync_result == "judge response"
    assert async_result == "judge response"
    assert len(mesh_client.requests) == 2
    assert mesh_client.requests[0].model == "openai/gpt-oss-120b"
    assert mesh_client.requests[0].messages[0].role == "user"
    assert mesh_client.requests[0].response_format == {"type": "json_object"}
    assert "api.meshapi.ai" not in mesh_client.requests[0].model
    assert model.get_model_name().startswith("Mesh/")
    assert os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] == "True"


class FakeMetric:
    def __init__(self, name):
        self.name = name
        self.score = 0.0

    def measure(self, test_case):
        output = test_case.actual_output
        self.score = 0.9 if output in {"edited draft", "learned draft"} else 0.5


def fake_metric_factory(_model):
    return [FakeMetric("hook_quality"), FakeMetric("brand_voice"), FakeMetric("relevance")]


def test_benchmark_scores_editor_and_memory_pairs_deterministically(tmp_path: Path):
    case_type, run_benchmarks, write_results = load_benchmark_modules()
    case = case_type(
        case_id="linkedin-ai-safety",
        brief="Write about AI safety for engineers.",
        brand_voice_profile={"tone_attributes": ["concise"]},
        drafts={
            "writer_initial": "initial draft",
            "editor_final": "edited draft",
            "cold_start": "cold draft",
            "learned": "learned draft",
        },
    )

    results = run_benchmarks([case], model=object(), metric_factory=fake_metric_factory)
    result_data = results[0].model_dump(mode="json")
    output_path = tmp_path / "results.json"
    write_results(results, output_path)

    assert set(result_data["variants"]) == {
        "writer_initial",
        "editor_final",
        "cold_start",
        "learned",
    }
    assert (
        result_data["variants"]["editor_final"]["composite_score"]
        > result_data["variants"]["writer_initial"]["composite_score"]
    )
    assert (
        result_data["variants"]["learned"]["composite_score"]
        > result_data["variants"]["cold_start"]["composite_score"]
    )
    assert json.loads(output_path.read_text(encoding="utf-8"))[0] == result_data
    assert "api_key" not in output_path.read_text(encoding="utf-8").lower()


def test_evaluation_case_requires_all_paired_outputs():
    case_type, _, _ = load_benchmark_modules()

    with pytest.raises(ValueError, match="variants"):
        case_type(
            case_id="incomplete",
            brief="A brief",
            brand_voice_profile={},
            drafts={"writer_initial": "only one output"},
        )


def test_real_geval_metrics_use_mesh_adapter_without_network():
    pytest.importorskip("deepeval", reason="install the optional eval extra")
    case_type, run_benchmarks, _ = load_benchmark_modules()
    from eval.mesh_model import MeshDeepEvalModel

    class StructuredJudgeMeshClient:
        def __init__(self):
            self.calls = []

        def complete(self, request):
            self.calls.append(request)
            return SimpleNamespace(
                content=(
                    '{"steps":["Check the hook and evidence."],"score":0.8,'
                    '"reason":"Evidence supports this score."}'
                )
            )

    mesh_client = StructuredJudgeMeshClient()
    model = MeshDeepEvalModel(mesh_client, model_name="openai/gpt-oss-120b")
    case = case_type(
        case_id="actual-geval-mock-transport",
        brief="A technical post about measurable AI safety.",
        brand_voice_profile={"tone_attributes": ["concise"]},
        learned_playbook=[{"rule": "Lead with a measurable outcome"}],
        drafts={
            variant: f"Draft variant: {variant}"
            for variant in (
                "writer_initial",
                "editor_final",
                "cold_start",
                "learned",
            )
        },
    )

    results = run_benchmarks([case], model)

    assert len(results) == 1
    assert set(results[0].variants) == set(case.drafts)
    assert mesh_client.calls
    assert all(call.model == "openai/gpt-oss-120b" for call in mesh_client.calls)
    assert all("user" in [message.role for message in call.messages] for call in mesh_client.calls)


def test_cli_without_mesh_key_writes_no_results(tmp_path: Path, monkeypatch):
    from app.config import Settings
    from eval.benchmark import main

    input_path = tmp_path / "cases.json"
    output_path = tmp_path / "results.json"
    input_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        "app.config.Settings",
            lambda: Settings(
                llm_provider="mesh",
                mesh_api_key=None,
                _env_file=None,
            ),
    )

    with pytest.raises(SystemExit):
        main(["--input", str(input_path), "--output", str(output_path)])

    assert not output_path.exists()


def _run_benchmark_cli(tmp_path, monkeypatch, settings, extra_args=()):
    from eval import benchmark

    input_path = tmp_path / "cases.json"
    output_path = tmp_path / "results.json"
    input_path.write_text("[]", encoding="utf-8")
    observed = {}
    monkeypatch.setattr("app.config.Settings", lambda: settings)
    monkeypatch.setattr(
        benchmark,
        "run_benchmarks",
        lambda cases, model: observed.update(model=model) or [],
    )
    monkeypatch.setattr(benchmark, "write_results", lambda results, path: None)

    benchmark.main(["--input", str(input_path), "--output", str(output_path), *extra_args])

    return observed["model"]


def test_benchmark_cli_uses_groq_model_by_default(tmp_path: Path, monkeypatch):
    from pydantic import SecretStr

    from app.config import Settings

    settings = Settings(
        llm_provider="groq",
        mesh_api_key=SecretStr("unused-mesh-key"),
        groq_api_key=SecretStr("test-groq-key"),
        groq_model="test-groq-model",
        _env_file=None,
    )

    model = _run_benchmark_cli(tmp_path, monkeypatch, settings)

    assert model._model_name == "test-groq-model"


def test_benchmark_cli_allows_an_explicit_model_override(tmp_path: Path, monkeypatch):
    from pydantic import SecretStr

    from app.config import Settings

    settings = Settings(
        llm_provider="groq",
        mesh_api_key=SecretStr("unused-mesh-key"),
        groq_api_key=SecretStr("test-groq-key"),
        groq_model="test-groq-model",
        _env_file=None,
    )

    model = _run_benchmark_cli(
        tmp_path,
        monkeypatch,
        settings,
        ["--model", "explicit-groq-model"],
    )

    assert model._model_name == "explicit-groq-model"


def test_benchmark_cli_rejects_missing_selected_provider_key(tmp_path: Path, monkeypatch):
    from pydantic import SecretStr

    from app.config import Settings
    from eval.benchmark import main

    input_path = tmp_path / "cases.json"
    output_path = tmp_path / "results.json"
    input_path.write_text("[]", encoding="utf-8")
    settings = Settings(
        llm_provider="groq",
        mesh_api_key=SecretStr("unused-mesh-key"),
        groq_api_key=None,
        groq_model="test-groq-model",
        _env_file=None,
    )
    monkeypatch.setattr("app.config.Settings", lambda: settings)
    monkeypatch.setattr(
        "eval.benchmark.run_benchmarks",
        lambda *_args, **_kwargs: pytest.fail("benchmark must not run without Groq credentials"),
    )

    with pytest.raises(SystemExit):
        main(["--input", str(input_path), "--output", str(output_path)])

    assert not output_path.exists()
