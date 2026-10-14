import asyncio
import os

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "True"

from deepeval.models import DeepEvalBaseLLM

from app.llm.mesh_client import LLMRequest, MeshClient


class MeshDeepEvalModel(DeepEvalBaseLLM):
    def __init__(self, mesh_client: MeshClient, model_name: str):
        self._mesh_client = mesh_client
        self._model_name = model_name
        super().__init__(model=model_name)

    def load_model(self):
        return self._mesh_client

    def generate(self, prompt: str, schema=None, **_kwargs) -> str:
        response = self._mesh_client.complete(
            LLMRequest(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0,
                response_format={"type": "json_object"} if schema is not None else None,
            )
        )
        if response.content is None:
            raise RuntimeError("Mesh evaluator returned empty content")
        return response.content

    async def a_generate(self, prompt: str, schema=None, **_kwargs) -> str:
        return await asyncio.to_thread(self.generate, prompt, schema)

    def get_model_name(self) -> str:
        return f"Mesh/{self._model_name}"
