"""영화 리뷰 감성 분석 웹앱 (Streamlit 프론트엔드). 데이터는 전부 백엔드에서 관리."""

import html
import pandas as pd
import streamlit as st

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import api_client

COLS = 4  # 한 줄에 표시할 영화 카드 수
KST = ZoneInfo("Asia/Seoul")
# 감성 라벨 → (표시 이름, 색상). 무채색 톤에 맞춘 저채도 색.
SENTIMENT = {
    "positive": ("긍정", "#7FC8A0"),
    "neutral": ("중립", "#A1A1AA"),
    "negative": ("부정", "#E08A8A"),
}
COLOR_BY_NAME = {name: color for name, color in SENTIMENT.values()}
CSS = """
<style>
.poster-wrap {position: relative; overflow: hidden; border-radius: 8px;
              aspect-ratio: 2 / 3; background: #1A1A1A;}
.poster {width: 100%; height: 100%; object-fit: cover; display: block;
         transition: transform .25s ease;}
.poster-wrap:hover .poster {transform: scale(1.04);}
.rating {position: absolute; top: 8px; left: 8px; padding: 2px 9px;
         border-radius: 999px; background: rgba(0, 0, 0, .7);
         color: #FFFFFF; font-size: .8rem; font-weight: 600;}
.m-title {font-size: 1.1rem; font-weight: 600; line-height: 1.3;
          height: 2.6em; margin: .6rem 0 .2rem; color: #FFFFFF;
          overflow: hidden; display: -webkit-box;
          -webkit-line-clamp: 2; -webkit-box-orient: vertical;}
.m-meta {font-size: .8rem; line-height: 1.4; height: 4.2em;
         color: #A1A1AA; overflow: hidden;}
.badge {display: inline-block; padding: 1px 9px; border-radius: 999px;
        font-size: .78rem; font-weight: 600; border: 1px solid;}
.stButton button {padding: .1rem .7rem; min-height: 0; font-size: .8rem;
                  border-color: #2E2E2E;}
</style>
"""

st.set_page_config(page_title="Movie Review", page_icon="🎬", layout="wide")


def to_kst(iso: str) -> str:
    """서버의 ISO 시각(UTC)을 KST 문자열로 변환한다.

    Args:
        iso: ISO 8601 문자열. tz 정보가 없으면(SQLite) UTC로 간주.

    Returns:
        str: 'YYYY-MM-DD HH:MM:SS' 형식의 KST 시각.
    """
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(KST).strftime("%Y-%m-%d %H:%M:%S")


def sentiment_text(review: dict) -> str:
    """감성 결과를 '라벨 점수' 문자열로 만든다.

    Args:
        review: 백엔드 ReviewRead 응답.

    Returns:
        str: 예) '긍정 4.53'.
    """
    label = review["sentiment_label"]
    name = SENTIMENT.get(label, (label, ""))[0]
    return f"{name} {review['sentiment_score']:.2f}"


def sentiment_badge(review: dict) -> str:
    """감성 결과를 색 배지 HTML로 만든다.

    Args:
        review: 백엔드 ReviewRead 응답.

    Returns:
        str: unsafe_allow_html 마크다운에 넣을 span 문자열.
    """
    color = SENTIMENT.get(review["sentiment_label"], ("", "#A1A1AA"))[1]
    return (
        f'<span class="badge" style="color:{color};border-color:{color}">'
        f"{html.escape(sentiment_text(review))}</span>"
    )


def sentiment_cell_style(value: str) -> str:
    """표의 감성 셀 글자색을 반환한다.

    Args:
        value: '긍정 4.53' 형태의 셀 값.

    Returns:
        str: CSS 선언 문자열.
    """
    return f"color: {COLOR_BY_NAME.get(value.split()[0], '#FFFFFF')}"


def render_add_movie_form() -> None:
    """사이드바에 영화 추가 폼을 그리고, 제출 시 백엔드에 등록한다."""
    with st.sidebar.form("add_movie", clear_on_submit=True):
        st.header("영화 추가")
        title = st.text_input("제목")
        release_date = st.date_input(
            "개봉일", value=date.today(), min_value=date(1900, 1, 1)
        )
        director = st.text_input("감독")
        genre = st.text_input("장르")
        poster_url = st.text_input("포스터 URL", placeholder="https://...")
        submitted = st.form_submit_button("등록")

    if not submitted:
        return
    fields = (title, director, genre, poster_url)
    if not all(f.strip() for f in fields):
        st.sidebar.error("모든 항목을 입력하세요.")
        return
    try:
        api_client.create_movie(
            {
                "title": title.strip(),
                "release_date": release_date.isoformat(),
                "director": director.strip(),
                "genre": genre.strip(),
                "poster_url": poster_url.strip(),
            }
        )
    except api_client.ApiError as e:
        st.sidebar.error(f"등록 실패: {e}")
        return
    st.toast(f"'{title.strip()}' 등록 완료")


def render_card(movie: dict) -> None:
    """영화 카드 1개(포스터+평점 배지, 제목, 정보, 삭제 버튼)를 그린다.

    Args:
        movie: 백엔드 MovieRead 응답.
    """
    title = html.escape(movie["title"])
    poster = html.escape(movie["poster_url"], quote=True)
    meta = html.escape(
        f"{movie['director']} · {movie['genre']} · {movie['release_date']}"
    )
    rating = movie["avg_rating"]
    rating_text = f"★ {rating:.2f}" if rating is not None else "★ -"
    with st.container(border=True):
        st.markdown(
            '<div class="poster-wrap">'
            f'<img class="poster" src="{poster}" alt="{title}">'
            f'<span class="rating">{rating_text}</span></div>'
            f'<div class="m-title" title="{title}">{title}</div>'
            f'<div class="m-meta">{meta}</div>',
            unsafe_allow_html=True,
        )
        if st.button("삭제", key=f"del_movie_{movie['id']}"):
            try:
                api_client.delete_movie(movie["id"])
            except api_client.ApiError as e:
                st.error(f"삭제 실패: {e}")
                return
            st.rerun()


def render_movies(movies: list[dict]) -> None:
    """영화 목록을 COLS개씩 그리드로 표시한다.

    Args:
        movies: 영화 목록.
    """
    if not movies:
        st.info("등록된 영화가 없습니다. 사이드바에서 추가하세요.")
        return
    for start in range(0, len(movies), COLS):
        row = movies[start:start + COLS]
        for col, movie in zip(st.columns(COLS), row):
            with col:
                render_card(movie)


def render_add_review_form(titles: dict[int, str]) -> None:
    """리뷰 등록 폼. 등록 후 평점이 갱신되도록 rerun한다.

    Args:
        titles: 영화 ID → 제목.
    """
    with st.form("add_review", clear_on_submit=True):
        st.subheader("리뷰 작성")
        movie_id = st.selectbox("영화", titles, format_func=titles.get)
        author = st.text_input("작성자")
        content = st.text_area("내용")
        submitted = st.form_submit_button("등록")

    if not submitted:
        return
    if not (author.strip() and content.strip()):
        st.error("작성자와 내용을 입력하세요.")
        return
    try:
        review = api_client.create_review(
            movie_id, {"author": author.strip(), "content": content.strip()}
        )
    except api_client.ApiError as e:
        st.error(f"등록 실패: {e}")
        return
    st.session_state["flash"] = f"리뷰 등록 완료 · {sentiment_text(review)}"
    st.rerun()


def render_recent_reviews(titles: dict[int, str]) -> None:
    """최근 리뷰 10개를 표로 표시한다.

    Args:
        titles: 영화 ID → 제목.
    """
    st.subheader("최근 리뷰 10개")
    reviews = api_client.list_recent_reviews(10)
    if not reviews:
        st.info("등록된 리뷰가 없습니다.")
        return
    df = pd.DataFrame(
        {
            "영화 ID": r["movie_id"],
            "영화": titles.get(r["movie_id"], "-"),
            "등록일": to_kst(r["created_at"]),
            "내용": r["content"],
            "감성": sentiment_text(r),
        }
        for r in reviews
    )
    styler = df.style
    # pandas 2.1+ 는 Styler.map, 이전 버전은 applymap
    apply = getattr(styler, "map", None) or styler.applymap
    st.dataframe(
        apply(sentiment_cell_style, subset=["감성"]), hide_index=True
    )


def render_movie_reviews(titles: dict[int, str]) -> None:
    """선택한 영화의 리뷰를 최신순으로 보여주고 삭제 버튼을 둔다.

    Args:
        titles: 영화 ID → 제목.
    """
    st.subheader("영화별 리뷰")
    movie_id = st.selectbox(
        "영화 선택", titles, format_func=titles.get, key="review_movie"
    )
    reviews = api_client.list_movie_reviews(movie_id)
    if not reviews:
        st.info("이 영화의 리뷰가 없습니다.")
        return
    for r in reviews:
        with st.container(border=True):
            body, action = st.columns([6, 1])
            body.markdown(
                f"**{html.escape(r['author'])}** &nbsp; "
                f"{sentiment_badge(r)}",
                unsafe_allow_html=True,
            )
            body.write(r["content"])
            body.caption(to_kst(r["created_at"]))
            if action.button("삭제", key=f"del_review_{r['id']}"):
                api_client.delete_review(r["id"])
                st.rerun()


def main() -> None:
    """페이지 전체를 그린다."""
    if flash := st.session_state.pop("flash", None):
        st.toast(flash)

    st.markdown(CSS, unsafe_allow_html=True)
    render_add_movie_form()  # 목록보다 먼저 처리해야 같은 실행에서 바로 반영됨
    st.title("🎬 영화 리뷰 감성 분석")
    try:
        with st.spinner("불러오는 중... (서버가 잠들어 있으면 최대 1분)"):
            movies = api_client.list_movies()
    except api_client.ApiError as e:
        st.error(f"백엔드 연결 실패: {e}")
        return

    tab_movies, tab_reviews = st.tabs(["영화 목록", "리뷰"])
    with tab_movies:
        render_movies(movies)
    with tab_reviews:
        if not movies:
            st.info("영화를 먼저 등록하세요.")
            return
        titles = {m["id"]: m["title"] for m in movies}
        try:
            render_add_review_form(titles)
            render_recent_reviews(titles)
            render_movie_reviews(titles)
        except api_client.ApiError as e:
            st.error(f"리뷰 처리 실패: {e}")


main()
