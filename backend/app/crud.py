"""DB 접근 로직. 라우터는 이 모듈만 호출한다."""

from sqlalchemy import Select, delete, func, select
from sqlalchemy.orm import Session

from app import sentiment
from app.models import Movie, Review
from app.schemas import MovieCreate, MovieRead, RatingRead, ReviewCreate


def _movie_with_rating() -> Select:
    """영화 + 평균 평점을 한 번의 LEFT JOIN/GROUP BY로 조회하는 쿼리."""
    return (
        select(Movie, func.avg(Review.sentiment_score))
        .outerjoin(Review, Review.movie_id == Movie.id)
        .group_by(Movie.id)
    )


def _to_read(movie: Movie, avg: float | None) -> MovieRead:
    """ORM 객체와 평균값을 응답 스키마로 변환한다."""
    read = MovieRead.model_validate(movie)
    read.avg_rating = None if avg is None else round(float(avg), 2)
    return read


def create_movie(db: Session, data: MovieCreate) -> MovieRead:
    """영화를 등록한다.

    Args:
        db: DB 세션.
        data: 등록할 영화 정보.

    Returns:
        MovieRead: 등록된 영화 (avg_rating=None).
    """
    movie = Movie(**data.model_dump())
    db.add(movie)
    db.commit()
    return _to_read(movie, None)


def list_movies(db: Session) -> list[MovieRead]:
    """전체 영화를 평균 평점과 함께 조회한다 (쿼리 1회, N+1 없음).

    Args:
        db: DB 세션.

    Returns:
        list[MovieRead]: id 오름차순 영화 목록.
    """
    rows = db.execute(_movie_with_rating().order_by(Movie.id)).all()
    return [_to_read(movie, avg) for movie, avg in rows]


def get_movie(db: Session, movie_id: int) -> MovieRead | None:
    """영화 단건을 평균 평점과 함께 조회한다.

    Args:
        db: DB 세션.
        movie_id: 영화 ID.

    Returns:
        MovieRead | None: 없으면 None.
    """
    row = db.execute(_movie_with_rating().where(Movie.id == movie_id)).first()
    return _to_read(*row) if row else None


def delete_movie(db: Session, movie_id: int) -> bool:
    """영화를 삭제한다. 리뷰는 DB의 ON DELETE CASCADE로 함께 삭제된다.

    Args:
        db: DB 세션.
        movie_id: 영화 ID.

    Returns:
        bool: 삭제되었으면 True, 대상이 없으면 False.
    """
    result = db.execute(delete(Movie).where(Movie.id == movie_id))
    db.commit()
    return result.rowcount > 0


def _newest_first(stmt: Select) -> Select:
    """최신순 정렬. created_at이 같으면 id로 순서를 고정한다."""
    return stmt.order_by(Review.created_at.desc(), Review.id.desc())


def create_review(db: Session, movie_id: int, data: ReviewCreate) -> Review | None:
    """리뷰를 등록하고 감성 분석 결과를 함께 저장한다.

    Args:
        db: DB 세션.
        movie_id: 영화 ID.
        data: 작성자, 내용.

    Returns:
        Review | None: 등록된 리뷰. 영화가 없으면 None.
    """
    if db.get(Movie, movie_id) is None:
        return None
    label, score = sentiment.analyze(data.content)
    review = Review(
        movie_id=movie_id,
        **data.model_dump(),
        sentiment_label=label,
        sentiment_score=score,
    )
    db.add(review)
    db.commit()
    db.refresh(review)  # DB가 채운 created_at 로드
    return review


def list_movie_reviews(db: Session, movie_id: int) -> list[Review] | None:
    """특정 영화의 리뷰를 최신순으로 조회한다.

    Args:
        db: DB 세션.
        movie_id: 영화 ID.

    Returns:
        list[Review] | None: 리뷰 목록. 영화가 없으면 None.
    """
    if db.get(Movie, movie_id) is None:
        return None
    stmt = _newest_first(select(Review).where(Review.movie_id == movie_id))
    return list(db.scalars(stmt))


def list_recent_reviews(db: Session, limit: int) -> list[Review]:
    """전체 리뷰를 최신순으로 limit개 조회한다 (created_at 인덱스 사용).

    Args:
        db: DB 세션.
        limit: 최대 개수.

    Returns:
        list[Review]: 리뷰 목록.
    """
    return list(db.scalars(_newest_first(select(Review)).limit(limit)))


def delete_review(db: Session, review_id: int) -> bool:
    """리뷰를 삭제한다.

    Args:
        db: DB 세션.
        review_id: 리뷰 ID.

    Returns:
        bool: 삭제되었으면 True, 대상이 없으면 False.
    """
    result = db.execute(delete(Review).where(Review.id == review_id))
    db.commit()
    return result.rowcount > 0


def get_rating(db: Session, movie_id: int) -> RatingRead | None:
    """영화의 평균 평점과 리뷰 수를 집계한다 (movie_id 인덱스 사용).

    Args:
        db: DB 세션.
        movie_id: 영화 ID.

    Returns:
        RatingRead | None: 집계 결과. 영화가 없으면 None.
    """
    if db.get(Movie, movie_id) is None:
        return None
    avg, count = db.execute(
        select(func.avg(Review.sentiment_score), func.count(Review.id)).where(
            Review.movie_id == movie_id
        )
    ).one()
    return RatingRead(
        movie_id=movie_id,
        avg_rating=None if avg is None else round(float(avg), 2),
        review_count=count,
    )
