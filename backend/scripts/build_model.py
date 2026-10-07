"""감성 모델 후보 비교 + ONNX 변환/동적 양자화. 로컬 PC 전용 (배포에 포함 안 됨).

사용 (backend/ 에서):
    python -m pip install -r requirements-dev.txt
    python scripts/build_model.py compare
    python scripts/build_model.py export daekeun-ml/koelectra-small-v3-nsmc
    python scripts/build_model.py calibrate

결과물: models/sentiment.int8.onnx, models/tokenizer.json,
        models/calibration.json (temperature)
"""

import json
import random
import statistics
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

CANDIDATES = [
    "daekeun-ml/koelectra-small-v3-nsmc",
    "monologg/koelectra-small-finetuned-nsmc",
    "monologg/koelectra-base-finetuned-nsmc",
]
NSMC_URL = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_{}.txt"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
SAMPLE_SIZE = 2000
MAX_LEN = 128
POSITIVE_INDEX = 1  # NSMC: 0=부정, 1=긍정
LATENCY_TEXT = "배우들 연기는 좋았는데 스토리가 너무 지루해서 아쉬웠다."


def load_nsmc_sample(
    split: str = "test", n: int = SAMPLE_SIZE
) -> tuple[list[str], np.ndarray]:
    """NSMC에서 고정 시드로 n개를 샘플링한다.

    Args:
        split: "train"(보정용) 또는 "test"(평가용).
        n: 샘플 수.

    Returns:
        tuple[list[str], np.ndarray]: (문장 목록, 라벨 배열).
    """
    with urllib.request.urlopen(NSMC_URL.format(split)) as resp:
        lines = resp.read().decode("utf-8").splitlines()[1:]  # 헤더 제외
    rows = [line.split("\t") for line in lines]
    rows = [(doc, int(label)) for _, doc, label in rows if doc.strip()]
    rows = random.Random(42).sample(rows, n)
    return [doc for doc, _ in rows], np.array([label for _, label in rows])


def median_ms(fn, runs: int = 50) -> float:
    """fn을 warmup 5회 후 runs회 실행한 지연시간 중앙값(ms).

    Args:
        fn: 인자 없는 함수.
        runs: 측정 횟수.

    Returns:
        float: 중앙값(ms).
    """
    for _ in range(5):
        fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def torch_predict(model, tokenizer, texts: list[str]) -> np.ndarray:
    """PyTorch 모델로 긍정 여부(0/1)를 배치 예측한다.

    Args:
        model: 분류 모델.
        tokenizer: 토크나이저.
        texts: 문장 목록.

    Returns:
        np.ndarray: 예측 라벨.
    """
    preds = []
    with torch.inference_mode():
        for i in range(0, len(texts), 64):
            batch = tokenizer(
                texts[i : i + 64],
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            preds.append(model(**batch).logits.argmax(-1).numpy())
    return np.concatenate(preds)


def ort_logits(session, tokenizer, texts: list[str]) -> np.ndarray:
    """ONNX 세션으로 배치 logits를 계산한다.

    Args:
        session: onnxruntime 세션.
        tokenizer: 토크나이저.
        texts: 문장 목록.

    Returns:
        np.ndarray: (N, 2) logits.
    """
    names = {i.name for i in session.get_inputs()}
    preds = []
    for i in range(0, len(texts), 64):
        batch = tokenizer(
            texts[i : i + 64],
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="np",
            return_token_type_ids=True,
        )
        feeds = {k: v.astype(np.int64) for k, v in batch.items() if k in names}
        preds.append(session.run(None, feeds)[0])
    return np.concatenate(preds)


def make_session(path: Path) -> ort.InferenceSession:
    """Render Free(CPU 1개 미만)와 비슷하게 단일 스레드 세션을 만든다.

    Args:
        path: ONNX 파일 경로.

    Returns:
        ort.InferenceSession: 추론 세션.
    """
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def compare() -> None:
    """후보 모델들의 정확도·크기·지연시간을 표로 출력한다."""
    torch.set_num_threads(1)
    texts, labels = load_nsmc_sample()
    print("| 모델 | 파라미터(M) | FP32 크기(MB) | NSMC 정확도 | 지연(ms) |")
    print("|---|---|---|---|---|")
    for name in CANDIDATES:
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name).eval()
        params = sum(p.numel() for p in model.parameters())
        acc = (torch_predict(model, tokenizer, texts) == labels).mean()
        single = tokenizer(LATENCY_TEXT, return_tensors="pt")
        with torch.inference_mode():
            ms = median_ms(lambda: model(**single))
        print(
            f"| {name} | {params / 1e6:.1f} | {params * 4 / 2**20:.0f} "
            f"| {acc:.4f} | {ms:.1f} |"
        )


def export_onnx(model, tokenizer, out_dir: Path) -> tuple[Path, Path]:
    """모델을 ONNX(FP32)로 변환한 뒤 INT8 동적 양자화한다.

    Args:
        model: 분류 모델 (eval 모드).
        tokenizer: fast 토크나이저.
        out_dir: 결과 저장 폴더.

    Returns:
        tuple[Path, Path]: (FP32 경로, INT8 경로).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32 = out_dir / "sentiment.fp32.onnx"
    int8 = out_dir / "sentiment.int8.onnx"
    dummy = tokenizer(LATENCY_TEXT, return_tensors="pt")
    # BERT/ELECTRA forward 인자 순서와 동일하게 맞춘다
    names = [n for n in ("input_ids", "attention_mask", "token_type_ids") if n in dummy]
    torch.onnx.export(
        model,
        tuple(dummy[n] for n in names),
        str(fp32),
        input_names=names,
        output_names=["logits"],
        dynamic_axes={
            **{n: {0: "batch", 1: "seq"} for n in names},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    return fp32, int8


def export(name: str) -> None:
    """선택한 모델을 변환·양자화하고 FP32/INT8 비교표를 출력한다.

    Args:
        name: Hugging Face 모델 ID.
    """
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).eval()
    with tempfile.TemporaryDirectory() as tmp:
        fp32, int8 = export_onnx(model, tokenizer, Path(tmp))
        MODEL_DIR.mkdir(exist_ok=True)
        int8_dst = MODEL_DIR / int8.name
        int8_dst.write_bytes(int8.read_bytes())
        tokenizer.backend_tokenizer.save(str(MODEL_DIR / "tokenizer.json"))
        print_onnx_report(tokenizer, fp32, int8_dst)
    print(f"\n저장 완료: {int8_dst}, {MODEL_DIR / 'tokenizer.json'}")


def print_onnx_report(tokenizer, fp32: Path, int8: Path) -> None:
    """FP32 vs INT8 ONNX의 크기·정확도·지연시간을 표로 출력한다.

    Args:
        tokenizer: 토크나이저.
        fp32: FP32 ONNX 경로.
        int8: INT8 ONNX 경로.
    """
    texts, labels = load_nsmc_sample()
    single = {
        k: v.astype(np.int64)
        for k, v in tokenizer(LATENCY_TEXT, return_tensors="np").items()
    }
    print("| 버전 | 크기(MB) | NSMC 정확도 | 지연(ms) |")
    print("|---|---|---|---|")
    for tag, path in (("ONNX FP32", fp32), ("ONNX INT8", int8)):
        session = make_session(path)
        names = {i.name for i in session.get_inputs()}
        feeds = {k: v for k, v in single.items() if k in names}
        logits = ort_logits(session, tokenizer, texts)
        acc = (logits.argmax(-1) == labels).mean()
        ms = median_ms(lambda: session.run(None, feeds))
        size = path.stat().st_size / 2**20
        print(f"| {tag} | {size:.1f} | {acc:.4f} | {ms:.1f} |")


def positive_prob(logits: np.ndarray, temperature: float) -> np.ndarray:
    """온도 보정 softmax로 긍정 확률을 계산한다.

    Args:
        logits: (N, 2) logits.
        temperature: 온도 T (>1이면 확률이 완만해짐).

    Returns:
        np.ndarray: (N,) 긍정 확률.
    """
    z = logits / temperature
    z = z - z.max(-1, keepdims=True)
    exp = np.exp(z)
    return exp[:, POSITIVE_INDEX] / exp.sum(-1)


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """이진 negative log-likelihood.

    Args:
        probs: 긍정 확률.
        labels: 정답 (0/1).

    Returns:
        float: 평균 NLL.
    """
    p = np.clip(np.where(labels == 1, probs, 1 - probs), 1e-12, 1)
    return float(-np.log(p).mean())


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """Expected Calibration Error (예측 신뢰도와 실제 정확도의 차이).

    Args:
        probs: 긍정 확률.
        labels: 정답 (0/1).
        bins: 구간 수.

    Returns:
        float: ECE (0에 가까울수록 잘 보정됨).
    """
    conf = np.maximum(probs, 1 - probs)
    correct = (probs >= 0.5) == (labels == 1)
    idx = np.minimum(((conf - 0.5) * 2 * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            gap = abs(conf[mask].mean() - correct[mask].mean())
            total += mask.mean() * gap
    return float(total)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """NLL을 최소화하는 온도 T를 그리드 탐색으로 찾는다.

    Args:
        logits: (N, 2) logits.
        labels: 정답 (0/1).

    Returns:
        float: 최적 T.
    """
    grid = np.round(np.arange(0.5, 10.01, 0.05), 2)
    losses = [nll(positive_prob(logits, t), labels) for t in grid]
    return float(grid[int(np.argmin(losses))])


def calibrate() -> None:
    """INT8 모델에  scaling을temperature 적용하고 보정 전후를 비교한다.

    T는 NSMC train 샘플로 찾고, 평가는 test 샘플로 한다 (데이터 누수 방지).

    Raises:
        FileNotFoundError: export를 먼저 실행하지 않았을 때.
    """
    session = make_session(MODEL_DIR / "sentiment.int8.onnx")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(MODEL_DIR / "tokenizer.json"), pad_token="[PAD]"
    )
    fit_texts, fit_labels = load_nsmc_sample("train")
    temperature = fit_temperature(ort_logits(session, tokenizer, fit_texts), fit_labels)
    texts, labels = load_nsmc_sample("test")
    logits = ort_logits(session, tokenizer, texts)
    print(f"최적 temperature T = {temperature}")
    print("| 버전 | 정확도 | NLL | ECE |")
    print("|---|---|---|---|")
    versions = (("보정 전", 1.0), ("보정 후", temperature))
    for tag, t in versions:
        probs = positive_prob(logits, t)
        acc = ((probs >= 0.5) == (labels == 1)).mean()
        print(
            f"| {tag} | {acc:.4f} | {nll(probs, labels):.4f} "
            f"| {ece(probs, labels):.4f} |"
        )
    (MODEL_DIR / "calibration.json").write_text(
        json.dumps({"temperature": temperature})
    )
    print(f"\n저장 완료: {MODEL_DIR / 'calibration.json'}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["compare"]:
        compare()
    elif sys.argv[1:2] == ["calibrate"]:
        calibrate()
    elif sys.argv[1:2] == ["export"] and len(sys.argv) == 3:
        export(sys.argv[2])
    else:
        sys.exit(__doc__)
