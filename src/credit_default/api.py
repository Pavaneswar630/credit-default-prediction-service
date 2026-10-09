"""FastAPI application exposing health and credit-default prediction."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from credit_default.service import PredictionService


class CreditApplication(BaseModel):
    """Raw borrower fields accepted from a prediction client."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    revolving_utilization: float | None = Field(alias="RevolvingUtilizationOfUnsecuredLines")
    age: float | None
    past_due_30_59: float | None = Field(alias="NumberOfTime30-59DaysPastDueNotWorse")
    debt_ratio: float | None = Field(alias="DebtRatio")
    monthly_income: float | None = Field(alias="MonthlyIncome")
    open_credit_lines: float | None = Field(alias="NumberOfOpenCreditLinesAndLoans")
    past_due_90: float | None = Field(alias="NumberOfTimes90DaysLate")
    real_estate_lines: float | None = Field(alias="NumberRealEstateLoansOrLines")
    past_due_60_89: float | None = Field(alias="NumberOfTime60-89DaysPastDueNotWorse")
    dependents: float | None = Field(alias="NumberOfDependents")


class PredictionResponse(BaseModel):
    """Prediction result with the deployed model version."""

    default_probability: float
    predicted_default: bool
    model_version: str


class HealthResponse(BaseModel):
    """Application and loaded-model readiness state."""

    status: str
    model_loaded: bool
    model_version: str | None = None


def create_app(model_path: str | None = None) -> FastAPI:
    """Create the app, optionally pointing it at a specific model artifact."""
    artifact_path = model_path or os.getenv(
        "MODEL_PATH", "models/credit_default.joblib"
    )
    predictor = PredictionService(artifact_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if predictor.artifact_path.is_file():
            predictor.load()
        app.state.predictor = predictor
        yield

    application = FastAPI(
        title="Credit Default Prediction Service",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse | JSONResponse:
        loaded = request.app.state.predictor.pipeline is not None
        response = HealthResponse(
            status="ok" if loaded else "model_unavailable",
            model_loaded=loaded,
            model_version=request.app.state.predictor.model_version,
        )
        if not loaded:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=response.model_dump(),
            )
        return response

    def get_predictor(request: Request) -> PredictionService:
        service = request.app.state.predictor
        if service.pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction model is not loaded",
            )
        return service

    @application.post("/predict", response_model=PredictionResponse)
    def predict(
        application_data: CreditApplication,
        predictor_service: Annotated[PredictionService, Depends(get_predictor)],
    ) -> PredictionResponse:
        raw_features = pd.DataFrame([application_data.model_dump(by_alias=True)])
        try:
            probability = predictor_service.predict(raw_features)
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return PredictionResponse(
            default_probability=probability,
            predicted_default=probability >= 0.5,
            model_version=predictor_service.model_version or "unknown",
        )

    return application


app = create_app()
