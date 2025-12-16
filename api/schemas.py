"""
Pydantic Schemas for API Request/Response Validation
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request schema for threat severity prediction."""
    
    features: List[float] = Field(
        ...,
        description="List of 41 feature values (NSL-KDD raw features)",
        min_items=41,
        max_items=41
    )
    
    include_explanation: bool = Field(
        default=False,
        description="Whether to include SHAP explanation"
    )
    
    @validator('features')
    def validate_features(cls, v):
        """Validate feature values."""
        if any(x != x for x in v):  # Check for NaN
            raise ValueError("Features cannot contain NaN values")
        return v


class FeatureContribution(BaseModel):
    """Feature contribution for explanation."""
    
    feature: str = Field(..., description="Feature name")
    contribution: float = Field(..., description="SHAP contribution value")


class PredictionResponse(BaseModel):
    """Response schema for threat severity prediction."""
    
    threat_score: float = Field(
        ...,
        description="Predicted threat severity score (0-10)",
        ge=0.0,
        le=10.0
    )
    
    risk_level: str = Field(
        ...,
        description="Risk level category (low/medium/high/critical)"
    )
    
    recommended_action: str = Field(
        ...,
        description="Recommended action based on risk level"
    )
    
    top_contributors: Optional[List[FeatureContribution]] = Field(
        default=None,
        description="Top contributing features (if explanation requested)"
    )
    
    inference_time_ms: float = Field(
        ...,
        description="Inference time in milliseconds"
    )
    
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Prediction timestamp"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status (healthy/unhealthy)")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: str = Field(..., description="Model version")
    last_update: Optional[str] = Field(None, description="Last model update time")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ExplainRequest(BaseModel):
    """Request schema for detailed explanation."""
    
    features: List[float] = Field(
        ...,
        description="List of 41 feature values (NSL-KDD raw features)",
        min_items=41,
        max_items=41
    )
    
    top_k: int = Field(
        default=10,
        description="Number of top features to return",
        ge=1,
        le=49
    )


class ExplainResponse(BaseModel):
    """Response schema for detailed explanation."""
    
    threat_score: float = Field(..., description="Predicted threat severity")
    
    feature_contributions: List[Dict[str, Any]] = Field(
        ...,
        description="Detailed feature contributions with values and SHAP values"
    )
    
    base_value: float = Field(
        ...,
        description="Base prediction value (model's expected value)"
    )
    
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Error timestamp"
    )
