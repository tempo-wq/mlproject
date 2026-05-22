"""
experiments/evaluate_summarizer.py
===================================
Оценка качества суммаризатора TF-IDF + TextRank на датасете формата .xlsx (Excel).

Метрики: ROUGE-1, ROUGE-2, ROUGE-L, Compression Ratio.
Baseline: Lead-N, Random-N.
"""

import sys
import random
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd  # <-- Добавили pandas для работы с Excel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from summarizer import Summarizer, split_sentences, preprocess_text

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# =====================================================================
# НАСТРОЙКИ ДАТАСЕТА EXCEL (МЕНЯТЬ ЗДЕСЬ)
# =====================================================================
DATASET_PATH = Path(__file__).parent / "dataset.xlsx" # Путь к Excel-файлу
TEXT_KEY = "text"             # Название КОЛОНКИ с полным текстом
SUMMARY_KEY = "reference_summary"       # Название КОЛОНКИ с эталонным саммари
VAL_SIZE = 45                 # Сколько примеров брать для подбора параметров (Grid Search)
TEST_SIZE = 100               # Сколько примеров брать для финального честного теста
MIN_WORDS = 13                # Отбрасывать слишком короткие тексты (защита от ошибок)
# =====================================================================

def load_dataset() -> List[Dict[str, str]]:
    """Загружает Excel-датасет через pandas и приводит к единому формату."""
    if not DATASET_PATH.exists():
        logger.error(f"Файл {DATASET_PATH} не найден!")
        logger.info(f"Убедитесь, что файл лежит в папке experiments/ и называется {DATASET_PATH.name}")
        sys.exit(1)

    logger.info(f"Чтение файла {DATASET_PATH.name}...")
    try:
        # Читаем Excel файл
        df = pd.read_excel(DATASET_PATH)
    except Exception as e:
        logger.error(f"Ошибка при чтении Excel: {e}")
        logger.info("Убедитесь, что установлена библиотека openpyxl (pip install openpyxl)")
        sys.exit(1)

    # Проверяем, есть ли нужные колонки
    if TEXT_KEY not in df.columns or SUMMARY_KEY not in df.columns:
        logger.error(f"Колонки '{TEXT_KEY}' или '{SUMMARY_KEY}' не найдены в таблице!")
        logger.info(f"Доступные колонки: {list(df.columns)}")
        sys.exit(1)

    # Очистка данных: удаляем строки, где нет текста или саммари (NaN)
    df = df.dropna(subset=[TEXT_KEY, SUMMARY_KEY])

    processed_data = []
    # Итерируемся по строкам датафрейма
    for _, row in df.iterrows():
        # Явно приводим к строке, чтобы избежать падений на числах
        text = str(row[TEXT_KEY]).strip()
        ref = str(row[SUMMARY_KEY]).strip()
        
        # Фильтрация битых или слишком коротких данных
        if len(text.split()) >= MIN_WORDS and len(ref.split()) > 3:
            processed_data.append({
                "text": text,
                "reference_summary": ref
            })
                
    logger.info(f"Загружено {len(processed_data)} валидных примеров после очистки NaN и фильтрации.")
    return processed_data


# ─────────────────────────────────────────────────────────────
# Вспомогательные функции метрик и бейзлайнов 
# ─────────────────────────────────────────────────────────────
def tokenize_for_rouge(text: str) -> List[str]:
    import re
    return re.findall(r"\b\w+\b", text.lower())

def get_ngrams(tokens: List[str], n: int) -> Dict[Tuple, int]:
    ngrams = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i : i + n])
        ngrams[gram] = ngrams.get(gram, 0) + 1
    return ngrams

def rouge_n(hypothesis: str, reference: str, n: int) -> Dict[str, float]:
    hyp_tokens = tokenize_for_rouge(hypothesis)
    ref_tokens = tokenize_for_rouge(reference)
    hyp_ngrams = get_ngrams(hyp_tokens, n)
    ref_ngrams = get_ngrams(ref_tokens, n)
    overlap = sum(min(hyp_ngrams.get(g, 0), ref_ngrams.get(g, 0)) for g in ref_ngrams)
    precision = overlap / max(sum(hyp_ngrams.values()), 1)
    recall = overlap / max(sum(ref_ngrams.values()), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1}

def lcs_length(x: List[str], y: List[str]) -> int:
    m, n = len(x), len(y)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]

def rouge_l(hypothesis: str, reference: str) -> Dict[str, float]:
    hyp_tokens = tokenize_for_rouge(hypothesis)
    ref_tokens = tokenize_for_rouge(reference)
    lcs = lcs_length(hyp_tokens, ref_tokens)
    precision = lcs / max(len(hyp_tokens), 1)
    recall = lcs / max(len(ref_tokens), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1}

def lead_n_summary(text: str, n: int) -> str:
    sentences = split_sentences(preprocess_text(text))
    return " ".join(sentences[:n]) if sentences else text

def random_n_summary(text: str, n: int, seed: int = 42) -> str:
    sentences = split_sentences(preprocess_text(text))
    if len(sentences) <= n:
        return text
    rng = random.Random(seed)
    chosen = sorted(rng.sample(range(len(sentences)), n))
    return " ".join(sentences[i] for i in chosen)


# ─────────────────────────────────────────────────────────────
# Grid Search
# ─────────────────────────────────────────────────────────────
@dataclass
class ExperimentConfig:
    compression_ratio: float
    similarity_threshold: float
    damping: float
    ngram_range: Tuple[int, int]

@dataclass
class ExperimentResult:
    config: ExperimentConfig
    avg_rouge1_f1: float = 0.0
    avg_rouge2_f1: float = 0.0
    avg_rougeL_f1: float = 0.0
    avg_compression: float = 0.0

def evaluate_config(config: ExperimentConfig, samples: List[Dict]) -> ExperimentResult:
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    summarizer = Summarizer(
        compression_ratio=config.compression_ratio,
        similarity_threshold=config.similarity_threshold,
        damping=config.damping,
    )
    summarizer.vectorizer = TfidfVectorizer(
        analyzer="word",
        tokenizer=lambda t: t.split(),
        lowercase=False,
        ngram_range=config.ngram_range,
        max_df=0.9,
        min_df=1,
        sublinear_tf=True,
    )

    r1_scores, r2_scores, rl_scores, comp_scores = [], [], [], []

    for sample in samples:
        try:
            summary = summarizer.summarize(sample["text"])
        except Exception:
            continue

        ref = sample["reference_summary"]
        r1_scores.append(rouge_n(summary, ref, 1)["f1"])
        r2_scores.append(rouge_n(summary, ref, 2)["f1"])
        rl_scores.append(rouge_l(summary, ref)["f1"])
        comp_scores.append(len(summary.split()) / max(len(sample["text"].split()), 1))

    if not r1_scores:
        return ExperimentResult(config=config)

    return ExperimentResult(
        config=config,
        avg_rouge1_f1=np.mean(r1_scores),
        avg_rouge2_f1=np.mean(r2_scores),
        avg_rougeL_f1=np.mean(rl_scores),
        avg_compression=np.mean(comp_scores),
    )

def run_grid_search(samples: List[Dict]) -> List[ExperimentResult]:
    configs = [
        ExperimentConfig(cr, st, d, ngr)
        for cr in [0.25, 0.35, 0.45]
        for st in [0.03, 0.05, 0.10]
        for d in [0.80, 0.85, 0.90]
        for ngr in [(1, 1), (1, 2)]
    ]

    results = []
    total = len(configs)
    for i, cfg in enumerate(configs):
        result = evaluate_config(cfg, samples)
        results.append(result)
        if (i + 1) % 10 == 0 or (i + 1) == total:
            logger.info(f"  [{i+1}/{total}] Проверено конфигураций...")

    return sorted(results, key=lambda r: r.avg_rouge1_f1, reverse=True)


# ─────────────────────────────────────────────────────────────
# Основной запуск
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  ОЦЕНКА КАЧЕСТВА СУММАРИЗАЦИИ НА EXCEL ДАННЫХ")
    print("=" * 70)

    data = load_dataset()
    if len(data) < 2:
        logger.error("Слишком мало данных для оценки. Проверьте Excel файл.")
        return

    # Перемешиваем и разбиваем датасет
    random.seed(42)
    random.shuffle(data)
    
    val_size_actual = min(VAL_SIZE, len(data) // 2)
    val_samples = data[:val_size_actual]
    test_samples = data[val_size_actual : val_size_actual + TEST_SIZE]

    print(f"\n[INFO] Разбиение: Val={len(val_samples)} (для GridSearch), Test={len(test_samples)} (для финала)")

    # ── 1. Grid Search на Val ──
    print("\n── 1. Grid Search (Поиск лучших гиперпараметров на Val) ──")
    results = run_grid_search(val_samples)
    best = results[0]
    
    print("\n  Топ-3 конфигурации (по ROUGE-1 F1):")
    for r in results[:3]:
        c = r.config
        print(f"  R1: {r.avg_rouge1_f1:.4f} | R-L: {r.avg_rougeL_f1:.4f} | CR: {c.compression_ratio}, ST: {c.similarity_threshold}, N-gram: {c.ngram_range}")

    print(f"\n✓ Победитель сохранен для финального теста!")

    # ── 2. Финальный честный тест на Test ──
    print("\n── 2. Финальное тестирование на отложенной выборке (Test) ──")
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    best_model = Summarizer(
        compression_ratio=best.config.compression_ratio,
        similarity_threshold=best.config.similarity_threshold,
        damping=best.config.damping
    )
    best_model.vectorizer = TfidfVectorizer(
        analyzer="word", tokenizer=lambda t: t.split(), lowercase=False,
        ngram_range=best.config.ngram_range, max_df=0.9, min_df=1, sublinear_tf=True
    )

    model_r1, model_rl = [], []
    lead_r1, lead_rl = [], []
    rand_r1, rand_rl = [], []

    for i, sample in enumerate(test_samples):
        if (i + 1) % 50 == 0:
            logger.info(f"  Обработано тестовых примеров: {i+1}/{len(test_samples)}")

        text = sample["text"]
        ref = sample["reference_summary"]

        try:
            our_summary = best_model.summarize(text)
        except Exception:
            continue

        n_target = max(1, len(split_sentences(our_summary)))
        lead_summary = lead_n_summary(text, n_target)
        rand_summary = random_n_summary(text, n_target)

        model_r1.append(rouge_n(our_summary, ref, 1)["f1"])
        model_rl.append(rouge_l(our_summary, ref)["f1"])
        
        lead_r1.append(rouge_n(lead_summary, ref, 1)["f1"])
        lead_rl.append(rouge_l(lead_summary, ref)["f1"])
        
        rand_r1.append(rouge_n(rand_summary, ref, 1)["f1"])
        rand_rl.append(rouge_l(rand_summary, ref)["f1"])

    print("\n── ИТОГОВЫЕ МЕТРИКИ (TEST SET) ──\n")
    print(f"  {'Алгоритм':<25} {'ROUGE-1 F1':>12} {'ROUGE-L F1':>12}")
    print(f"  {'-'*52}")
    print(f"  {'Наш TF-IDF+TextRank':<25} {np.mean(model_r1):>12.4f} {np.mean(model_rl):>12.4f}")
    print(f"  {'Lead-N (Baseline)':<25} {np.mean(lead_r1):>12.4f} {np.mean(lead_rl):>12.4f}")
    print(f"  {'Random-N (Baseline)':<25} {np.mean(rand_r1):>12.4f} {np.mean(rand_rl):>12.4f}")
    print("\n======================================================================")

if __name__ == "__main__":
    main()
