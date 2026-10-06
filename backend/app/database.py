"""DB 엔진/세션 설정. DATABASE_URL 하나로 SQLite(로컬) ↔ Neon(배포) 전환."""

import os
from collections.abc import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

"""
Set up this Neon project in the current working directory.

1. `npm i -g neon@latest && neon login`
2. `neon skills -y`
3. `neon mcp -y`
4. `neon link --project-id empty-cherry-16588921 --branch production -y`
5. `neon config init`
6. Update `neon.ts`:

```ts
import { defineConfig } from "@neon/config/v1";

export default defineConfig({});
```

7. `neon deploy`
"""

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")

# Neon은 postgres:// 또는 postgresql:// 로 제공 → psycopg(v3) 드라이버로 고정
for _prefix in ("postgres://", "postgresql://"):
    if DATABASE_URL.startswith(_prefix):
        DATABASE_URL = "postgresql+psycopg://" + DATABASE_URL[len(_prefix) :]
        break

IS_SQLITE = DATABASE_URL.startswith("sqlite")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if IS_SQLITE else {},
    pool_pre_ping=True,  # Neon scale-to-zero 이후 끊긴 커넥션 자동 감지
)

if IS_SQLITE:

    @event.listens_for(engine, "connect")
    def _enable_sqlite_fk(dbapi_conn, _record) -> None:
        """SQLite는 FK(ON DELETE CASCADE)가 기본 비활성이라 연결마다 켠다."""
        dbapi_conn.execute("PRAGMA foreign_keys=ON")


SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    """ORM 모델 공통 베이스."""


def get_db() -> Iterator[Session]:
    """요청 단위 DB 세션을 제공하는 FastAPI 의존성.

    Yields:
        Session: 요청 종료 시 자동으로 닫히는 세션.
    """
    with SessionLocal() as db:
        yield db
