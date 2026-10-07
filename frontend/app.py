"""영화 리뷰 감성 분석 웹앱 (Streamlit 프론트엔드). 데이터는 전부 백엔드에서 관리."""

from datetime import date

import api_client
import streamlit as st

COLS = 4  # 한 줄에 표시할 영화 카드 수
LABELS = {"positive": "😊 긍정", "negative": "😞 부정"}

st.set_page_config(page_title="Movie Review", page_icon="🎬", layout="wide")


def sentiment_text(review: dict) -> str:
    """감성 결과를 '라벨 (점수)' 문자열로 만든다.

    Args:
        review: 백엔드 ReviewRead 응답.

    Returns:
        str: 예) '😊 긍정 (4.53)'.
    """
    label = LABELS.get(review["sentiment_label"], review["sentiment_label"])
    return f"{label} ({review['sentiment_score']:.2f})"


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
    """영화 카드 1개(포스터, 제목, 평점, 정보, 삭제 버튼)를 그린다.

    Args:
        movie: 백엔드 MovieRead 응답.
    """
    with st.container(border=True):
        st.image(movie["poster_url"])
        st.subheader(movie["title"])
        rating = movie["avg_rating"]
        st.markdown(f"**★ {rating:.2f} / 5**" if rating is not None else "★ 평점 없음")
        st.caption(f"{movie['director']} · {movie['genre']} · {movie['release_date']}")
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
        row = movies[start : start + COLS]
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
    st.dataframe(
        [
            {
                "영화 ID": r["movie_id"],
                "영화": titles.get(r["movie_id"], "-"),
                "등록일": r["created_at"][:19].replace("T", " "),
                "내용": r["content"],
                "감성": sentiment_text(r),
            }
            for r in reviews
        ],
        hide_index=True,
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
            body.markdown(f"**{r['author']}** · {sentiment_text(r)}")
            body.write(r["content"])
            body.caption(r["created_at"][:19].replace("T", " "))
            if action.button("삭제", key=f"del_review_{r['id']}"):
                api_client.delete_review(r["id"])
                st.rerun()


def main() -> None:
    """페이지 전체를 그린다."""
    if flash := st.session_state.pop("flash", None):
        st.toast(flash)

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
