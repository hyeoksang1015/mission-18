"""리뷰 CRUD 라우터."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app import crud
from app.database import get_db
from app.schemas import Message, ReviewCreate, ReviewRead

router = APIRouter(tags=["Reviews"])

MOVIE_NOT_FOUND = {404: {"model": Message, "description": "영화가 존재하지 않음"}}
REVIEW_NOT_FOUND = {404: {"model": Message, "description": "리뷰가 존재하지 않음"}}


@router.post(
    "/movies/{movie_id}/reviews",
    response_model=ReviewRead,
    status_code=status.HTTP_201_CREATED,
    responses=MOVIE_NOT_FOUND,
    summary="리뷰 등록",
    description=(
        "영화에 리뷰를 등록합니다. 등록 시 감성 분석이 자동 실행되어 "
        "sentiment_label(긍정/부정)과 sentiment_score(긍정 확률×5)가 저장됩니다."
    ),
)
def create_review(
    movie_id: int, data: ReviewCreate, db: Session = Depends(get_db)
) -> ReviewRead:
    """리뷰 등록 엔드포인트.

    Raises:
        HTTPException: 영화가 없으면 404.
    """
    review = crud.create_review(db, movie_id, data)
    if review is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Movie not found")
    return review


@router.get(
    "/movies/{movie_id}/reviews",
    response_model=list[ReviewRead],
    responses=MOVIE_NOT_FOUND,
    summary="영화별 리뷰 조회",
    description="특정 영화의 리뷰를 최신순으로 반환합니다.",
)
def list_movie_reviews(
    movie_id: int, db: Session = Depends(get_db)
) -> list[ReviewRead]:
    """영화별 리뷰 조회 엔드포인트.

    Raises:
        HTTPException: 영화가 없으면 404.
    """
    reviews = crud.list_movie_reviews(db, movie_id)
    if reviews is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Movie not found")
    return reviews


@router.get(
    "/reviews",
    response_model=list[ReviewRead],
    summary="최근 리뷰 조회",
    description="전체 리뷰를 최신순으로 limit개(기본 10개) 반환합니다.",
)
def list_recent_reviews(
    limit: int = Query(10, ge=1, le=100, description="최대 개수"),
    db: Session = Depends(get_db),
) -> list[ReviewRead]:
    """최근 리뷰 조회 엔드포인트."""
    return crud.list_recent_reviews(db, limit)


@router.delete(
    "/reviews/{review_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=REVIEW_NOT_FOUND,
    summary="리뷰 삭제",
    description="리뷰 ID로 리뷰를 삭제합니다.",
)
def delete_review(review_id: int, db: Session = Depends(get_db)) -> None:
    """리뷰 삭제 엔드포인트.

    Raises:
        HTTPException: 리뷰가 없으면 404.
    """
    if not crud.delete_review(db, review_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Review not found")