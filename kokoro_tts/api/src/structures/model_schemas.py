"""Voice configuration schemas."""

from pydantic import BaseModel, Field


class VoiceConfig(BaseModel):
    """Voice configuration."""

    use_cache: bool = Field(True, description="Whether to cache loaded voices")
    cache_size: int = Field(3, description="Number of voices to cache")
    validate_on_load: bool = Field(
        True, description="Whether to validate voices when loading"
    )

    class Config:
        frozen = True  # Make config immutable
