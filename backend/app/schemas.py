"""요청/응답 Pydantic 스키마."""

from datetime import date
from pydantic import BaseModel, ConfigDict, Field


class MovieCreate(BaseModel):
    """영화 등록 요청."""

    title: str = Field(min_length=1, max_length=200, examples=["기생충"])
    release_date: date = Field(examples=["2019-05-30"])
    director: str = Field(min_length=1, max_length=100, examples=["봉준호"])
    genre: str = Field(min_length=1, max_length=50, examples=["드라마"])
    poster_url: str = Field(
        max_length=500,
        pattern=r"^https?://",
        examples=["https://example.com/poster.jpg"],
    )


class MovieRead(MovieCreate):
    """영화 응답. avg_rating은 리뷰가 없으면 null."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    avg_rating: float | None = Field(
        default=None, ge=0, le=5, description="리뷰 감성 점수 평균 (0~5)"
    )


class Message(BaseModel):
    """에러 응답."""

    detail: str