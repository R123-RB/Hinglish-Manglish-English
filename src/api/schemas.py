"""
schemas.py
==========
Pydantic models for the FastAPI translation API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class TranslationRequest(BaseModel):
    text: str = Field(
        ...,
        description="Code-mixed Hinglish / Manglish / Mixed sentence to translate.",
        example="Bro njan innu varilla, mood illa",
        min_length=1,
        max_length=512,
    )
    include_literal: bool = Field(
        default=False,
        description="If true, also return the literal (word-for-word) translation.",
    )
    refine_fluency: bool = Field(
        default=True,
        description="Apply post-processing fluency correction to the output.",
    )


class TranslationResponse(BaseModel):
    input:       str            = Field(..., description="Original input text")
    translation: str            = Field(..., description="Culturally accurate English")
    literal:     Optional[str]  = Field(None, description="Literal translation (if requested)")
    lang_script: str            = Field(..., description="Detected input script")
    lang_tags:   List[List]     = Field(..., description="Per-token language tags")
    emotion:     Optional[str]  = Field(None, description="Detected emotion (future)")


class BatchTranslationRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        description="List of code-mixed sentences to translate.",
        max_items=50,
    )
    refine_fluency: bool = Field(default=True)


class BatchTranslationResponse(BaseModel):
    results: List[TranslationResponse]
    total:   int


class HealthResponse(BaseModel):
    status:     str
    model:      str
    device:     str
    version:    str = "1.0.0"


class FeedbackRequest(BaseModel):
    input:       str
    translation: str
    meaning_ok:  bool
    tone_ok:     bool
    natural_ok:  bool
    correction:  Optional[str] = None
    notes:       Optional[str] = None
