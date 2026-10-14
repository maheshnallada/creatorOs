import asyncio
import os

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "True"

from deepeval.models import DeepEvalBaseLLM

from app.llm.mesh_client import LLMClient, LLMRequest


class ProviderDeepEvalModel(DeepEvalBaseLLM):
    def __init__(self, llm_client: LLMClient, model_name: str):
        self._llm_client = llm_client
        self._model_name = model_name
        self._provider = getattr(llm_client, "provider", "mesh")
        super().__init__(model=model_name)

    def load_model(self):
        return self._llm_client

    def generate(self, prompt: str, schema=None, **_kwargs) -> str:
        response = self._llm_client.complete(
            LLMRequest(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0,
                response_format={"type": "json_object"} if schema is not None else None,
            )
        )
        if response.content is None:
            raise RuntimeError(f"{self._provider.title()} evaluator returned empty content")
        return response.content

    async def a_generate(self, prompt: str, schema=None, **_kwargs) -> str:
        return await asyncio.to_thread(self.generate, prompt, schema)

    def get_model_name(self) -> str:
        return f"{self._provider.title()}/{self._model_name}"


MeshDeepEvalModel = ProviderDeepEvalModel
