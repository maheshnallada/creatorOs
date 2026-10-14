import json

from app.agents.contracts import (
    BrandVoiceProfile,
    BrandVoiceSampleInput,
    BrandVoiceSamplesRequest,
)
from app.agents.runtime import AgentRuntime
from app.config import PlatformAgentConfig


class BrandVoiceAgent:
    def __init__(self, runtime: AgentRuntime, config: PlatformAgentConfig):
        self._runtime = runtime
        self._config = config

    def build_profile(
        self,
        brand_id: str,
        samples: list[BrandVoiceSampleInput],
    ) -> BrandVoiceProfile:
        if not brand_id.strip():
            raise ValueError("brand_id must not be empty")
        request = BrandVoiceSamplesRequest(samples=samples)
        sample_data = [sample.model_dump() for sample in request.samples]
        return self._runtime.generate_json(
            self._config,
            system_prompt_vars={},
            instruction_prompt_vars={"samples_json": json.dumps(sample_data, ensure_ascii=False)},
            output_model=BrandVoiceProfile,
        )
