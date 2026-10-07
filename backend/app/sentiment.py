def analyze(text: str) -> tuple[str, float]:
    """리뷰 텍스트의 감성을 분석한다.

    Args:
        text : 리뷰 본문

    Returns:
        tuple[str, float]: (라벨, 점수) / 점수 = 긍정확률 x 5 (0-5)
    """

    return "pending", 2.5
