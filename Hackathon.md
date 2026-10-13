CreatorOS
Build a Self-Improving Content Studio

Mesh API is the official sponsor and using it is mandatory. Every LLM/AI call in your project must route through Mesh API. A submission that does not use Mesh API is not a valid submission.
The Challenge
Build CreatorOS, a multi-agent content studio where specialized agents collaborate to plan, write, and refine content, generate a matching visual, schedule it, and then learn from performance feedback to improve future output — all while staying true to a learned brand voice.

This is not a one-shot content generator. Tools where AI agents write on-brand content already exist. Their common limitation is that they are assistants, not closed-loop systems: a human approves the output, handles visuals separately, pushes it live, and nothing learns from how the content actually performed. The hard, valuable part of this challenge is the loop: generating content, capturing real performance feedback, and feeding that back so the system improves over time.

The Story (what you are building, end to end)
A user gives a brief: "a LinkedIn post about AI safety."

A strategist agent decides the angle, hook, and audience. A writer agent drafts the piece for that specific platform. An editor agent critiques the draft against concrete criteria and the writer revises it — this self-improvement step is where quality comes from. A visual agent generates a matching image. The finished piece is stored, queued for a time, and (bonus) posted. Performance feedback — real metrics if available, or user-provided or simulated engagement — is captured and fed back, so a learning step can adjust what the system produces next. A brand-voice profile learned from the user's past content keeps every piece consistent. Every important decision (the angle, the edits) is logged.

What You Will Build
1. Multi-agent orchestration (the backbone)

A coordinator that runs a pipeline of distinct agents and passes work between them.
At least three genuinely distinct agents (for example strategist, writer, editor, visual). Distinct means each has its own job and its own prompt or its own tools — not the same prompt behind different names.
2. The editor refinement loop (a core of this track)

An editor agent critiques the draft against concrete criteria and the writer revises it. The output must genuinely change as a result — not pass through untouched.
3. Brand voice

A profile learned from the user's existing content (more than a single sample) that is injected into generation so output stays on-voice. Different brand profiles produce visibly different output.
4. Visuals and scheduling

A matching image is generated for the content, and the finished piece is stored and queued for a scheduled time.
5. The performance feedback loop (the core of this track)

Performance data for published content — real, user-entered, or simulated — is captured and fed back so the system adjusts future content. This closed loop is what separates CreatorOS from a one-shot generator, and it is the heaviest-weighted part of the track.
6. An auditable decision trail

Every important decision is recorded with the reason behind it: the angle the strategist chose, the edits the editor asked for, and what the system learned from performance feedback. Each record is linked to the content piece it belongs to.
7. Self-evaluation (we reward this heavily)

Define a quality measure for your content and write a script that scores it, for example scoring drafts before and after the editor loop, or comparing pieces generated with and without the feedback learned so far. Record the numbers in the repo. A team that genuinely built this almost always measured whether it works.
Highlighted Bonus (optional, makes you stand out)
Structured agent framework (e.g. LangGraph) with named nodes and a real strategist → writer → editor loop, not a flat sequence.
Multi-platform adaptation: one idea adapted properly for different platforms and formats.
Real publishing through a platform API, with engagement read back into the system.
Real learning: a step that provably changes future output based on what performed best, not just stored numbers.
Suggested Stack (use what you are comfortable with)
Backend: Python (Flask or FastAPI)
Agents: LangGraph, CrewAI, AutoGen, or your own orchestrator
LLM access (mandatory): all LLM/AI calls must go through Mesh API, the official sponsor. Mesh is an OpenAI-compatible gateway (base_url="https://api.meshapi.ai/v1", key prefix rsk_): one key gives access to many models. Point the OpenAI SDK at Mesh and read your key from the MESH_API_KEY environment variable. Full guide: Mesh API docs.
Images: any image generation model you can reach through Mesh API
Database: SQLite or PostgreSQL
Frontend: a simple web UI or API (Streamlit, Gradio, Jinja2, or a JS frontend)
Scheduling: Celery, APScheduler, or cron
You may use user-entered or simulated engagement data if real platform metrics are not available. Keep your API keys in a local, gitignored .env. Since your code is what is evaluated, your credentials are never needed by us.

Submission Requirements
A public GitHub repository with all your code and a README explaining what you built, how the agents cooperate, and how to run it.
A short note on your key design decisions and your self-evaluation results (your quality measure and the numbers).
(Optional) A short demo video and a deployed URL, reviewed for shortlisted submissions.
What a Great Submission Looks Like
Distinct cooperating agents with a coordinator, and an editor whose critique visibly changes the draft.
A brand voice learned from real samples that genuinely changes what gets written.
Performance feedback that actually changes the next piece the system produces.
A visible decision trail and honest before and after numbers.
Build something you would be proud to show in an interview.

Using Mesh API (required)
Mesh API is mandatory — every LLM/AI call must go through it. Mesh is an OpenAI-compatible gateway: one key gives access to many models (GPT, Claude, Gemini, and more).

Create an API key (it starts with rsk_) on the Mesh dashboard, then point the OpenAI SDK at Mesh and read your key from the environment:

import os
from openai import OpenAI

client = OpenAI(base_url="https://api.meshapi.ai/v1", api_key=os.getenv("MESH_API_KEY"))
client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)
Add your key to a gitignored .env and as the GitHub secret MESH_API_KEY. Full guide: Mesh API docs.

Setup & Requirements
Follow this to make sure the automated checks pass and your project is eligible for evaluation.

Your repository must contain:

All your source code.
A requirements.txt (or pyproject.toml / Pipfile) listing a web framework (Flask or FastAPI) and the LLM client you use through Mesh API.
A README.md describing what you built, how your agents cooperate, your key design decisions, and your self-evaluation results.
A .gitignore that includes .env — never commit secrets.
Repository secrets (Settings → Secrets and variables → Actions) — new to GitHub secrets? Follow this short guide:

MESH_API_KEY — your Mesh API key. Mandatory; every LLM/AI call must go through Mesh.
SUBMISSION_TOKEN — your submission token, shown on your dashboard as soon as you register.
Your keys stay in your own repository and are only used by checks running in your own GitHub Actions — they are never sent to us.

Enabling the automated checks:

Download the workflow file: https://careerapi-production.krishnaik.in/api/ci/hackathons/agent-arena-2026/workflow.yml
Add it to your repository at .github/workflows/agent-arena-2026-checks.yml (create the .github/workflows/ folders if they don't exist).
Add the secrets listed above.
Push a commit. The checks run automatically — results appear in your repository's Actions tab and on your submission dashboard.
Critical checks (must pass for your commit to be eligible for evaluation):

Code compiles — all Python files are free of syntax errors.
Dependencies present — your requirements list a web framework and an LLM client.
Mesh API is used — your code routes LLM calls through Mesh (api.meshapi.ai base URL, or the meshapi SDK).
Mesh API key works — your MESH_API_KEY secret is a valid key. Verified with a metadata call that spends no credits.
Advisory checks (feedback only — these do not block you): no committed .env, a README is present, a .gitignore that ignores .env.

The automated checks are a first filter — passing them does not finalize your score. A failing critical check simply means "fix and push again"; it is not a penalty.

Mesh API — docs (mandatory LLM gateway)
LangGraph — agent orchestration
CrewAI
FastAPI
APScheduler — background jobs