# Evaluation Case Inputs

The benchmark consumes JSON objects matching `EvaluationCase` in `eval/benchmark.py`. Each case must contain actual saved draft outputs for all four paired variants: `writer_initial`, `editor_final`, `cold_start`, and `learned`, plus the brief, brand voice profile, and learned playbook.

Populate cases from real CreatorOS runs before invoking the live benchmark. Do not put generated sample scores or placeholder draft text in committed result artifacts. The offline unit tests use synthetic values only to verify runner calculations; they are not reported as benchmark results.

Run with the optional dependencies and credentials for the selected provider. Mesh is the default and must be used for hackathon/submission runs. For local Groq runs, set `APP_ENV=local`, `LLM_PROVIDER=groq`, `GROQ_API_KEY`, and `GROQ_MODEL` in `.env`.

```bash
python -m pip install -e '.[eval]'
python -m eval.benchmark --input eval/cases/my-run-pairs.json --output eval/results/phase3.json
```