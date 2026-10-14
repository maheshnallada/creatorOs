from fastapi import APIRouter, HTTPException, Request

from app.repositories.runs import RunNotFoundError
from app.services.learning import (
    BrandAssociationRequiredError,
    FeedbackInput,
    FeedbackResult,
    LearningServiceError,
)

router = APIRouter(prefix="/api/v1/runs", tags=["feedback"])


@router.post("/{run_id}/feedback", response_model=FeedbackResult)
def record_feedback(run_id: str, feedback: FeedbackInput, request: Request):
    learning_service = request.app.state.learning_service
    if learning_service is None:
        raise HTTPException(status_code=503, detail="Learning service is unavailable")
    if request.app.state.run_repository.get(run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")
    try:
        return learning_service.record_feedback(run_id, feedback)
    except RunNotFoundError:
        raise HTTPException(status_code=404, detail="Run not found") from None
    except BrandAssociationRequiredError:
        raise HTTPException(status_code=409, detail="Run has no brand association") from None
    except LearningServiceError:
        raise HTTPException(
            status_code=502,
            detail="Performance feedback was recorded, but learning failed.",
        ) from None
