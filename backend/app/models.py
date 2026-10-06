"""ORM 모델. Alembic 미사용이므로 reviews 테이블도 D1에 함께 확정한다."""

from datetime import date, datetime

from app.database import Base
from sqlalchemy import Date, DateTime, Float, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column


class Movie(Base):
    """영화 정보."""

    __tablename__ = "movies"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    release_date: Mapped[date] = mapped_column(Date)
    director: Mapped[str] = mapped_column(String(100))
    genre: Mapped[str] = mapped_column(String(50))
    poster_url: Mapped[str] = mapped_column(String(500))


class Review(Base):
    """리뷰 + 감성 분석 결과. sentiment_score = 긍정 확률 × 5 (0~5)."""

    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(primary_key=True)
    movie_id: Mapped[int] = mapped_column(
        ForeignKey("movies.id", ondelete="CASCADE"), index=True
    )
    author: Mapped[str] = mapped_column(String(50))
    content: Mapped[str] = mapped_column(Text)
    sentiment_label: Mapped[str] = mapped_column(String(10))
    sentiment_score: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
