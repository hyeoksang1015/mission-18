"""백엔드(FastAPI) 호출 모듈. 화면 코드는 이 모듈만 사용한다."""

import os

import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 90  # Render Free spin-down 후 재기동(약 1분) 대기

_session = requests.Session()  # 커넥션 재사용


class ApiError(Exception):
    """백엔드 연결 실패 또는 4xx/5xx 응답."""


def _request(method: str, path: str, **kwargs) -> dict | list | None:
    """백엔드에 요청을 보내고 JSON 본문을 반환한다.

    Args:
        method: HTTP 메서드.
        path: `/movies` 형태의 경로.
        **kwargs: requests에 그대로 전달할 인자 (json 등).

    Returns:
        dict | list | None: 응답 JSON. 본문이 없으면(204) None.

    Raises:
        ApiError: 연결 실패 또는 에러 응답.
    """
    try:
        resp = _session.request(
            method, f"{BACKEND_URL}{path}", timeout=TIMEOUT, **kwargs
        )
    except requests.RequestException as e:
        raise ApiError(f"서버에 연결할 수 없습니다 ({BACKEND_URL})") from e
    if not resp.ok:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise ApiError(f"{resp.status_code}: {detail}")
    return resp.json() if resp.content else None


def list_movies() -> list[dict]:
    """전체 영화를 평균 평점과 함께 조회한다.

    Returns:
        list[dict]: 영화 목록.
    """
    return _request("GET", "/movies")


def create_movie(data: dict) -> dict:
    """영화를 등록한다.

    Args:
        data: title, release_date(ISO 문자열), director, genre, poster_url.

    Returns:
        dict: 등록된 영화.
    """
    return _request("POST", "/movies", json=data)


def delete_movie(movie_id: int) -> None:
    """영화를 삭제한다 (리뷰도 함께 삭제).

    Args:
        movie_id: 영화 ID.
    """
    _request("DELETE", f"/movies/{movie_id}")


def create_review(movie_id: int, data: dict) -> dict:
    """리뷰를 등록한다 (백엔드에서 감성 분석 자동 실행).

    Args:
        movie_id: 영화 ID.
        data: author, content.

    Returns:
        dict: 감성 분석 결과가 포함된 리뷰.
    """
    return _request("POST", f"/movies/{movie_id}/reviews", json=data)


def list_movie_reviews(movie_id: int) -> list[dict]:
    """특정 영화의 리뷰를 최신순으로 조회한다.

    Args:
        movie_id: 영화 ID.

    Returns:
        list[dict]: 리뷰 목록.
    """
    return _request("GET", f"/movies/{movie_id}/reviews")


def list_recent_reviews(limit: int = 10) -> list[dict]:
    """전체 리뷰를 최신순으로 limit개 조회한다.

    Args:
        limit: 최대 개수.

    Returns:
        list[dict]: 리뷰 목록.
    """
    return _request("GET", "/reviews", params={"limit": limit})


def delete_review(review_id: int) -> None:
    """리뷰를 삭제한다.

    Args:
        review_id: 리뷰 ID.
    """
    _request("DELETE", f"/reviews/{review_id}")
