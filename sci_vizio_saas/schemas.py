"""API request/response models."""

import uuid
from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

UNFILED = "unfiled"


class PlanOut(BaseModel):
    id: str
    name: str
    price_monthly_usd: float
    max_documents: int
    max_file_mb: int
    max_figures_per_document: int


class UsageOut(BaseModel):
    documents: int
    figures: int
    storage_bytes: int


class MeOut(BaseModel):
    id: uuid.UUID
    email: Optional[str]
    display_name: Optional[str]
    plan: PlanOut
    usage: UsageOut
    subscription_status: Optional[str]
    current_period_end: Optional[datetime]
    billing_enabled: bool
    created_at: datetime


class MeUpdate(BaseModel):
    display_name: Optional[str] = Field(default=None, max_length=200)


class CollectionIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=2000)

    @field_validator("name")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Name must not be blank")
        return v


class CollectionUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=2000)


class CollectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: Optional[str]
    document_count: int = 0
    created_at: datetime


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    title: Optional[str]
    collection_id: Optional[uuid.UUID]
    status: str
    error: Optional[str]
    size_bytes: int
    page_count: Optional[int]
    figure_count: int
    skipped_figure_count: int
    created_at: datetime
    processed_at: Optional[datetime]


class DocumentPage(BaseModel):
    items: List[DocumentOut]
    total: int


class UploadRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=500)
    size_bytes: int = Field(gt=0)
    content_type: Literal["application/pdf"] = "application/pdf"
    sha256: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    collection_id: Optional[uuid.UUID] = None


class PresignedUploadOut(BaseModel):
    url: str
    method: str
    headers: Dict[str, str]


class UploadResponse(BaseModel):
    document: DocumentOut
    upload: PresignedUploadOut


class DocumentUpdate(BaseModel):
    title: Optional[str] = Field(default=None, max_length=1000)
    collection_id: Optional[uuid.UUID] = None


class UrlOut(BaseModel):
    url: str


class FigureOut(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    page: int
    figure_type: str
    image_type: Optional[str]
    title: Optional[str]
    description: Optional[str]
    analysis: dict
    width: int
    height: int
    image_url: str


class DocumentRef(BaseModel):
    id: uuid.UUID
    filename: str
    title: Optional[str]


class CollectionRef(BaseModel):
    id: uuid.UUID
    name: str


class SearchHit(FigureOut):
    score: Optional[float] = None
    document: DocumentRef
    collection: Optional[CollectionRef]


class SearchRequest(BaseModel):
    query: str = Field(default="", max_length=500)
    # Collection ids, or "unfiled" for documents outside any collection.
    collections: List[str] = Field(default_factory=list, max_length=100)
    documents: List[uuid.UUID] = Field(default_factory=list, max_length=100)
    figure_types: List[str] = Field(default_factory=list, max_length=50)
    # With no figure_types selected, decorative images (icons, logos...) are hidden unless this is set.
    include_decorative: bool = False
    limit: int = Field(default=24, ge=1, le=60)
    offset: int = Field(default=0, ge=0, le=10_000)

    @field_validator("collections")
    @classmethod
    def _valid_collections(cls, values: List[str]) -> List[str]:
        for v in values:
            if v != UNFILED:
                uuid.UUID(v)
        return values


class FacetValue(BaseModel):
    value: str
    label: str
    count: int


class Facets(BaseModel):
    figure_types: List[FacetValue]
    collections: List[FacetValue]
    documents: List[FacetValue]


class SearchResponse(BaseModel):
    total: int
    results: List[SearchHit]
    facets: Facets
