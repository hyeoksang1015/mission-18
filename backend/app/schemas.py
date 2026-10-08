"""요청/응답 Pydantic 스키마."""

from datetime import date, datetime

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


class MovieUpdate(BaseModel):
    """영화 수정 요청. 보낸 필드만 수정된다 (부분 수정)."""

    title: str | None = Field(default=None, min_length=1, max_length=200)
    release_date: date | None = None
    director: str | None = Field(default=None, min_length=1, max_length=100)
    genre: str | None = Field(default=None, min_length=1, max_length=50)
    poster_url: str | None = Field(
        default=None,
        max_length=500,
        pattern=r"^https?://",
        examples=["https://example.com/new-poster.jpg"],
    )


class MovieRead(MovieCreate):
    """영화 응답. avg_rating은 리뷰가 없으면 null."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    avg_rating: float | None = Field(
        default=None, ge=0, le=5, description="리뷰 감성 점수 평균 (0~5)"
    )


class ReviewCreate(BaseModel):
    """리뷰 등록 요청. 영화 ID는 경로로 받는다."""

    author: str = Field(min_length=1, max_length=50, examples=["홍길동"])
    content: str = Field(
        min_length=1, max_length=1000, examples=["배우들 연기가 정말 좋았어요."]
    )


class ReviewRead(ReviewCreate):
    """리뷰 응답 (감성 분석 결과 포함)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    movie_id: int
    sentiment_label: str = Field(description="positive / neutral / negative")
    sentiment_score: float = Field(ge=0, le=5, description="긍정 확률 × 5")
    created_at: datetime


class RatingRead(BaseModel):
    """영화 평점 응답."""

    movie_id: int
    avg_rating: float | None = Field(
        ge=0, le=5, description="리뷰 감성 점수 평균 (0~5), 리뷰 없으면 null"
    )
    review_count: int


class Message(BaseModel):
    """에러 응답."""

    detail: str
