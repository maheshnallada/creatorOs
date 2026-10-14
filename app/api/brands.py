from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.agents.contracts import BrandVoiceProfile, BrandVoiceSamplesRequest
from app.models.memory import BrandVoiceProfileVersion

router = APIRouter(prefix="/api/v1/brands", tags=["brands"])


class BrandProfileResponse(BaseModel):
    brand_id: str
    version: int
    profile: dict[str, Any]
    source_sample_ids: list[str]
    source_sample_count: int


def serialize_profile(version: BrandVoiceProfileVersion) -> BrandProfileResponse:
    return BrandProfileResponse(
        brand_id=version.brand_id,
        version=version.version,
        profile=version.profile,
        source_sample_ids=version.source_sample_ids,
        source_sample_count=len(version.source_sample_ids),
    )


@router.post("/{brand_id}/samples", response_model=BrandProfileResponse)
def add_brand_samples(
    brand_id: str,
    request_body: BrandVoiceSamplesRequest,
    request: Request,
):
    if not brand_id.strip():
        raise HTTPException(status_code=422, detail="brand_id must not be empty")
    repository = request.app.state.brand_memory_repository
    brand_voice_agent = request.app.state.brand_voice_agent
    if repository is None or brand_voice_agent is None:
        raise HTTPException(status_code=503, detail="Brand voice service is unavailable")

    try:
        profile: BrandVoiceProfile = brand_voice_agent.build_profile(brand_id, request_body.samples)
    except Exception:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail="Brand voice extraction failed; the active profile was not changed.",
        ) from None

    _, version = repository.create_profile_from_samples(
        brand_id,
        [sample.model_dump() for sample in request_body.samples],
        profile.model_dump(),
    )
    return serialize_profile(version)


@router.get("/{brand_id}/profile", response_model=BrandProfileResponse)
def get_brand_profile(brand_id: str, request: Request):
    repository = request.app.state.brand_memory_repository
    if repository is None:
        raise HTTPException(status_code=503, detail="Brand voice service is unavailable")
    profile = repository.get_active_profile(brand_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Brand profile not found")
    return serialize_profile(profile)
