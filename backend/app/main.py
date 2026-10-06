"""FastAPI 앱 진입점. 실행: uvicorn app.main:app --reload (backend/ 에서)."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.database import Base, engine
from app.routers import movies


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """앱 시작 시 테이블 생성 (D4에서 감성 모델 1회 로드도 여기에 추가)."""
    Base.metadata.create_all(engine)
    yield


app = FastAPI(
    title="Movie Review Sentiment API",
    description="영화 정보, 리뷰, 리뷰 감성 분석 API",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(movies.router)


@app.get("/health", summary="헬스체크", description="서버 기동 여부 확인용.")
def health() -> dict[str, str]:
    """헬스체크 엔드포인트."""
    return {"status": "ok"}