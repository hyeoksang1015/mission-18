"""리뷰 감성 분석. ONNX Runtime + tokenizers만 사용 (배포본에 PyTorch 없음)."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MAX_LEN = 128
POSITIVE_INDEX = 1  # NSMC: 0=부정, 1=긍정
NEUTRAL_BAND = (0.4, 0.6)  # 보정된 긍정 확률이 이 범위면 중립


@lru_cache(maxsize=1)
def load() -> tuple[ort.InferenceSession, Tokenizer, set[str], float]:
    """모델·토크나이저·보정 온도를 1회 로드한다 (lifespan에서 미리 적재).

    Returns:
        tuple: (세션, 토크나이저, 모델 입력 이름 집합, temperature).

    Raises:
        FileNotFoundError: models/ 에 모델 파일이 없을 때.
    """
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1  # Render Free CPU 제한
    session = ort.InferenceSession(
        str(MODEL_DIR / "sentiment.int8.onnx"),
        opts,
        providers=["CPUExecutionProvider"],
    )
    tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    tokenizer.enable_truncation(MAX_LEN)
    tokenizer.no_padding()
    calib = MODEL_DIR / "calibration.json"
    temperature = (
        json.loads(calib.read_text())["temperature"] if calib.exists() else 1.0
    )
    names = {i.name for i in session.get_inputs()}
    return session, tokenizer, names, temperature


def analyze(text: str) -> tuple[str, float]:
    """리뷰 텍스트의 감성을 분석한다.

    Args:
        text: 리뷰 본문.

    Returns:
        tuple[str, float]: (positive/neutral/negative, 보정된 긍정 확률 × 5).
    """
    session, tokenizer, names, temperature = load()
    enc = tokenizer.encode(text)
    feeds = {
        "input_ids": [enc.ids],
        "attention_mask": [enc.attention_mask],
        "token_type_ids": [enc.type_ids],
    }
    feeds = {k: np.array(v, dtype=np.int64) for k, v in feeds.items() if k in names}
    logits = session.run(None, feeds)[0][0] / temperature
    exp = np.exp(logits - logits.max())  # 안정적인 softmax
    pos = float(exp[POSITIVE_INDEX] / exp.sum())
    low, high = NEUTRAL_BAND
    label = "positive" if pos >= high else "negative" if pos <= low else "neutral"
    return label, round(pos * 5, 4)
