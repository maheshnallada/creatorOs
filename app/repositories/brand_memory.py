from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.orm import sessionmaker

from app.models.memory import BrandVoiceProfileVersion, BrandVoiceSample


class BrandMemoryRepository:
    def __init__(self, session_factory: sessionmaker):
        self._session_factory = session_factory

    @staticmethod
    def _new_samples(brand_id: str, samples: list[dict[str, Any]]) -> list[BrandVoiceSample]:
        if not brand_id.strip():
            raise ValueError("brand_id must not be empty")
        normalized_samples = []
        seen_bodies = set()
        for sample in samples:
            body = sample.get("body", "").strip()
            if not body:
                raise ValueError("sample body must not be empty")
            normalized_body = " ".join(body.split()).casefold()
            if normalized_body in seen_bodies:
                raise ValueError("brand samples must be distinct")
            seen_bodies.add(normalized_body)
            normalized_samples.append(
                BrandVoiceSample(
                    brand_id=brand_id,
                    body=body,
                    source_label=sample.get("source_label"),
                )
            )
        return normalized_samples

    def add_samples(
        self,
        brand_id: str,
        samples: list[dict[str, Any]],
    ) -> list[BrandVoiceSample]:
        normalized_samples = self._new_samples(brand_id, samples)
        with self._session_factory.begin() as session:
            session.add_all(normalized_samples)
        return normalized_samples

    def create_profile_from_samples(
        self,
        brand_id: str,
        samples: list[dict[str, Any]],
        profile: dict[str, Any],
    ) -> tuple[list[BrandVoiceSample], BrandVoiceProfileVersion]:
        normalized_samples = self._new_samples(brand_id, samples)
        if len(normalized_samples) < 3:
            raise ValueError("brand voice profiles require at least three source samples")

        with self._session_factory.begin() as session:
            session.add_all(normalized_samples)
            session.flush()
            profile_version = self._save_profile_in_session(
                session,
                brand_id,
                profile,
                [sample.id for sample in normalized_samples],
            )
        return normalized_samples, profile_version

    def list_samples(self, brand_id: str) -> list[BrandVoiceSample]:
        with self._session_factory() as session:
            return list(
                session.scalars(
                    select(BrandVoiceSample)
                    .where(BrandVoiceSample.brand_id == brand_id)
                    .order_by(BrandVoiceSample.created_at, BrandVoiceSample.id)
                )
            )

    def save_profile(
        self,
        brand_id: str,
        profile: dict[str, Any],
        source_sample_ids: list[str],
    ) -> BrandVoiceProfileVersion:
        unique_sample_ids = list(dict.fromkeys(source_sample_ids))
        if len(unique_sample_ids) < 3:
            raise ValueError("brand voice profiles require at least three source samples")

        with self._session_factory.begin() as session:
            profile_version = self._save_profile_in_session(
                session, brand_id, profile, unique_sample_ids
            )
        return profile_version

    @staticmethod
    def _save_profile_in_session(
        session,
        brand_id: str,
        profile: dict[str, Any],
        source_sample_ids: list[str],
    ) -> BrandVoiceProfileVersion:
        samples = list(
            session.scalars(
                select(BrandVoiceSample).where(
                    BrandVoiceSample.brand_id == brand_id,
                    BrandVoiceSample.id.in_(source_sample_ids),
                )
            )
        )
        if len(samples) != len(source_sample_ids):
            raise ValueError("source samples must belong to the requested brand")

        current_version = (
            session.scalar(
                select(func.max(BrandVoiceProfileVersion.version)).where(
                    BrandVoiceProfileVersion.brand_id == brand_id
                )
            )
            or 0
        )
        session.execute(
            update(BrandVoiceProfileVersion)
            .where(BrandVoiceProfileVersion.brand_id == brand_id)
            .values(is_active=False)
        )
        profile_version = BrandVoiceProfileVersion(
            brand_id=brand_id,
            version=current_version + 1,
            profile=profile,
            source_sample_ids=source_sample_ids,
            is_active=True,
        )
        session.add(profile_version)
        return profile_version

    def get_active_profile(self, brand_id: str) -> BrandVoiceProfileVersion | None:
        with self._session_factory() as session:
            return session.scalar(
                select(BrandVoiceProfileVersion).where(
                    BrandVoiceProfileVersion.brand_id == brand_id,
                    BrandVoiceProfileVersion.is_active.is_(True),
                )
            )

    def list_profile_versions(self, brand_id: str) -> list[BrandVoiceProfileVersion]:
        with self._session_factory() as session:
            return list(
                session.scalars(
                    select(BrandVoiceProfileVersion)
                    .where(BrandVoiceProfileVersion.brand_id == brand_id)
                    .order_by(BrandVoiceProfileVersion.version)
                )
            )
