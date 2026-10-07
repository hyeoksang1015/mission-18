"""시연용 데이터 시딩. 영화 4개 + 영화당 리뷰 12개를 API로 등록한다.

API를 거치므로 입력 검증과 감성 분석이 실제 서비스와 동일하게 실행된다.
같은 제목(공백·콜론 무시)의 영화가 이미 있으면 새로 만들지 않고 리뷰만 추가한다.

사용 (backend/ 에서):
    python scripts/seed.py                    # 로컬 (http://127.0.0.1:8000)
    BACKEND_URL=https://xxx.onrender.com python scripts/seed.py
    python scripts/seed.py --reset-reviews    # 대상 영화의 기존 리뷰 삭제 후 등록
"""

import json
import os
import sys
import urllib.error
import urllib.request

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 90  # Render 재기동 대기
PLACEHOLDER = "https://placehold.co/300x450/png?text={}"

MOVIES = [
    {
        "title": "레지던트 이블 : 0번째 밤",
        "release_date": "2026-09-17",
        "director": "잭 크레거",
        "genre": "공포, SF",
        "poster_url": PLACEHOLDER.format("Resident+Evil"),
        "reviews": [
            ("호러매니아", "처음부터 끝까지 긴장감이 장난 아니다. 정말 무서웠어요"),
            ("게임원작팬", "원작 게임 분위기를 잘 살렸다. 팬으로서 너무 만족"),
            ("심야상영", "사운드가 너무 좋아서 극장에서 보길 잘했다"),
            ("공포초보", "무서웠지만 재밌어서 끝까지 눈을 못 뗐어요"),
            ("장르덕후", "오랜만에 제대로 된 공포 영화를 본 느낌. 최고"),
            ("팝콘러버", "러닝타임 내내 몰입해서 봤다. 강력 추천"),
            ("스릴추구", "분위기와 연출이 훌륭하다. 다음 편도 기대된다"),
            ("친구랑관람", "친구랑 소리 지르면서 봤는데 너무 재밌었어요 ㅋㅋ"),
            ("깜짝놀람싫어", "놀래키는 장면만 많고 이야기는 너무 허술했다"),
            ("원작존중", "원작이랑 너무 달라서 실망했어요. 별로"),
            ("지루함", "중반부터 늘어져서 지루했다. 돈이 아깝다"),
            ("반반", "분위기는 좋은데 결말이 좀 아쉬웠다"),
        ],
    },
    {
        "title": "오디세이",
        "release_date": "2026-08-05",
        "director": "크리스토퍼 놀란",
        "genre": "모험, 판타지",
        "poster_url": PLACEHOLDER.format("Odyssey"),
        "reviews": [
            ("놀란팬", "역시 놀란 감독. 스케일이 압도적이다"),
            ("아이맥스필수", "아이맥스로 꼭 보세요. 영상미가 정말 대단합니다"),
            ("고전문학", "고전 서사시를 이렇게 웅장하게 옮기다니 감탄했다"),
            ("음악감상", "음악이 장면마다 감정을 끌어올린다. 소름 돋았어요"),
            ("모험좋아", "세 시간 가까이 되는데 전혀 지루하지 않았다"),
            ("배우팬", "배우들 연기가 하나같이 훌륭했다. 최고의 영화"),
            ("재관람", "두 번 봤는데 두 번째가 더 좋았어요"),
            ("영화광", "올해 본 영화 중 가장 인상 깊었다. 강력 추천"),
            ("졸린관객", "너무 길고 느려서 중간에 졸았다"),
            ("기대이하", "기대를 너무 많이 해서 그런지 실망스러웠다"),
            ("이해불가", "이야기가 복잡하고 산만해서 몰입이 안 됐어요"),
            ("취향차이", "잘 만든 건 알겠지만 내 취향은 아니었다"),
        ],
    },
    {
        "title": "러너",
        "release_date": "2026-09-07",
        "director": "Scott Waugh",
        "genre": "액션, 코미디",
        "poster_url": PLACEHOLDER.format("Runner"),
        "reviews": [
            ("액션러버", "액션이 시원시원하고 속도감이 대단하다. 너무 재밌음"),
            ("웃음보장", "웃기고 신나고 다 갖췄다. 스트레스가 확 풀렸어요"),
            ("팝콘필수", "생각 없이 즐기기 딱 좋은 영화. 강추"),
            ("케미좋아", "두 주인공 케미가 정말 좋았다 ㅋㅋ"),
            ("주말관객", "주말에 가볍게 보기 좋았어요. 만족합니다"),
            ("폭발덕후", "폭발 장면 스케일이 엄청나다. 극장에서 봐야 함"),
            ("두번본사람", "두 번 봤는데 두 번 다 재밌게 봤어요"),
            ("가족관람", "가족이랑 다 같이 웃으면서 봤습니다. 추천해요"),
            ("논리파", "개연성이 너무 없어서 보는 내내 답답했다"),
            ("유머코드", "개그가 너무 유치해서 하나도 안 웃겼다"),
            ("뻔한전개", "뻔한 전개라 금방 지루해졌어요. 별로"),
            ("무덤덤", "그럭저럭 볼만했지만 기억에 남지는 않는다"),
        ],
    },
    {
        "title": "스파이더맨: 브랜드 뉴 데이",
        "release_date": "2026-07-29",
        "director": "데스틴 대니얼 크레턴",
        "genre": "액션, SF",
        "poster_url": PLACEHOLDER.format("Spider-Man"),
        "reviews": [
            ("마블팬", "역시 스파이더맨. 처음부터 끝까지 너무 재밌었다"),
            ("거미줄액션", "도심 액션 장면이 정말 화려하고 시원하다"),
            ("오랜팬", "오래 기다린 보람이 있었다. 감동적이었어요"),
            ("청춘감성", "주인공 성장 이야기가 잘 그려져서 좋았다"),
            ("극장필수", "큰 화면으로 봐야 하는 영화. 강력 추천합니다"),
            ("쿠키필수", "쿠키 영상까지 완벽했다. 다음 편이 너무 기대됨"),
            ("팝콘러버", "시간 가는 줄 모르고 봤어요. 최고"),
            ("재관람", "벌써 세 번 봤어요. 볼 때마다 재밌다"),
            ("히어로피로", "히어로 영화가 이제 너무 뻔하게 느껴진다. 실망"),
            ("스토리중시", "액션만 화려하고 이야기는 너무 허술했다"),
            ("기대이하", "예고편이 제일 재밌었다. 별로였어요"),
            ("반반", "액션은 좋았는데 후반부 전개가 좀 아쉬웠다"),
        ],
    },
]


def normalize(title: str) -> str:
    """제목 비교용 정규화 (공백·콜론 제거).

    Args:
        title: 영화 제목.

    Returns:
        str: 정규화된 제목.
    """
    return "".join(c for c in title if not c.isspace() and c != ":")


def call(method: str, path: str, body: dict | None = None):
    """백엔드 API를 호출하고 JSON 응답을 반환한다.

    Args:
        method: HTTP 메서드.
        path: `/movies` 형태의 경로.
        body: JSON 본문.

    Returns:
        dict | list | None: 응답 JSON (204면 None).

    Raises:
        SystemExit: 4xx/5xx 응답 (실패한 요청과 응답 본문을 출력).
    """
    data = None if body is None else json.dumps(body).encode()
    url = BACKEND_URL + path
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise SystemExit(f"{method} {url} -> {e.code}: {detail}") from e
    return json.loads(raw) if raw else None


def seed(reset_reviews: bool) -> None:
    """영화와 리뷰를 등록하고 영화별 결과를 출력한다.

    Args:
        reset_reviews: True면 대상 영화의 기존 리뷰를 먼저 삭제한다.
    """
    print(f"대상 서버: {BACKEND_URL}")
    existing = {normalize(m["title"]): m["id"] for m in call("GET", "/movies")}
    print("| 영화 | 등록 리뷰 | 긍정 | 중립 | 부정 | 평균 평점 |")
    print("|---|---|---|---|---|---|")
    for movie in MOVIES:
        fields = {k: v for k, v in movie.items() if k != "reviews"}
        movie_id = existing.get(normalize(movie["title"]))
        if movie_id is None:
            movie_id = call("POST", "/movies", fields)["id"]
        elif reset_reviews:
            for review in call("GET", f"/movies/{movie_id}/reviews"):
                call("DELETE", f"/reviews/{review['id']}")
        counts = {"positive": 0, "neutral": 0, "negative": 0}
        for author, content in movie["reviews"]:
            review = call(
                "POST",
                f"/movies/{movie_id}/reviews",
                {"author": author, "content": content},
            )
            counts[review["sentiment_label"]] += 1
        rating = call("GET", f"/movies/{movie_id}/rating")
        print(
            f"| {movie['title']} | {len(movie['reviews'])} "
            f"| {counts['positive']} | {counts['neutral']} "
            f"| {counts['negative']} | {rating['avg_rating']} |"
        )


if __name__ == "__main__":
    seed(reset_reviews="--reset-reviews" in sys.argv[1:])
