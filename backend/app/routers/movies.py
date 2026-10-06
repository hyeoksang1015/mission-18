"""영화 CRUD 라우터."""

from sqlalchemy.orm import Session

from app import crud
from app.database import get_db
from app.schemas import Message, MovieCreate, MovieRead
from fastapi import APIRouter, Depends, HTTPException, status

router = APIRouter(prefix="/movies", tags=["Movies"])

NOT_FOUND = {404: {"model": Message, "description": "영화가 존재하지 않음"}}


@router.post(
    "",
    response_model=MovieRead,
    status_code=status.HTTP_201_CREATED,
    summary="영화 등록",
    description="제목, 개봉일, 감독, 장르, 포스터 URL로 영화를 등록합니다.",
)
def create_movie(data: MovieCreate, db: Session = Depends(get_db)) -> MovieRead:
    """영화 등록 엔드포인트."""
    return crud.create_movie(db, data)


@router.get(
    "",
    response_model=list[MovieRead],
    summary="전체 영화 조회",
    description="등록된 모든 영화를 평균 평점(리뷰 감성 점수 평균, 0~5)과 함께 반환합니다.",
)
def list_movies(db: Session = Depends(get_db)) -> list[MovieRead]:
    """전체 영화 조회 엔드포인트."""
    return crud.list_movies(db)


@router.get(
    "/{movie_id}",
    response_model=MovieRead,
    responses=NOT_FOUND,
    summary="특정 영화 조회",
    description="영화 ID로 단건을 평균 평점과 함께 조회합니다.",
)
def get_movie(movie_id: int, db: Session = Depends(get_db)) -> MovieRead:
    """특정 영화 조회 엔드포인트.

    Raises:
        HTTPException: 영화가 없으면 404.
    """
    movie = crud.get_movie(db, movie_id)
    if movie is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Movie not found")
    return movie


@router.delete(
    "/{movie_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=NOT_FOUND,
    summary="특정 영화 삭제",
    description="영화를 삭제합니다. 해당 영화의 리뷰도 함께 삭제됩니다(CASCADE).",
)
def delete_movie(movie_id: int, db: Session = Depends(get_db)) -> None:
    """특정 영화 삭제 엔드포인트.

    Raises:
        HTTPException: 영화가 없으면 404.
    """
    if not crud.delete_movie(db, movie_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Movie not found")
