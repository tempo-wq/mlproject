"""
experiments/evaluate_summarizer.py
===================================
Оценка качества суммаризатора TF-IDF + TextRank.

Метрики:
- ROUGE-1, ROUGE-2, ROUGE-L (стандарт для суммаризации)
- Compression Ratio
- Baseline: первые N предложений ("Lead-N")
- Baseline: случайные N предложений

Запуск:
    python experiments/evaluate_summarizer.py

Зависимости:
    pip install rouge-score
"""

import sys
import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from summarizer import Summarizer, split_sentences, preprocess_text

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Тестовые данные — примеры транскрибированных голосовых
# (имитация реальных входных данных бота)
# ─────────────────────────────────────────────────────────────

TEST_SAMPLES = [
    {
        "id": "sample_01",
        "source": "transcribed_voice",
        "text": (
            "Сегодня я хотел рассказать про встречу которая у нас была. "
            "Мы обсуждали проект по разработке нового приложения. "
            "Основная задача состоит в том чтобы создать удобный интерфейс для пользователей. "
            "Команда согласилась что нужно сначала провести исследование рынка. "
            "Потом на основании этого исследования выбрать технологии. "
            "Дедлайн по первому этапу поставили на конец следующего месяца. "
            "Ответственным назначили Алексея. "
            "Он должен подготовить презентацию с результатами анализа конкурентов. "
            "Следующая встреча запланирована на пятницу в три часа дня."
        ),
        "reference_summary": (
            "Обсуждался проект по разработке приложения с удобным интерфейсом. "
            "Решено провести исследование рынка, затем выбрать технологии. "
            "Алексей готовит анализ конкурентов, дедлайн — конец следующего месяца."
        ),
    },
    {
        "id": "sample_02",
        "source": "transcribed_voice",
        "text": (
            "Привет хочу рассказать про то что случилось сегодня утром. "
            "Я ехал на работу и попал в большую пробку на кольцевой. "
            "Простоял там почти час вместо обычных двадцати минут. "
            "В итоге опоздал на совещание где обсуждали квартальный отчёт. "
            "Коллеги рассказали что на встрече решили перенести сдачу отчёта на следующую неделю. "
            "Это связано с тем что несколько человек ещё не сдали свои части. "
            "Теперь крайний срок это пятница следующей недели."
        ),
        "reference_summary": (
            "Из-за пробки опоздал на совещание по квартальному отчёту. "
            "Сдачу отчёта перенесли на следующую пятницу — не все успели сдать свои части."
        ),
    },
    {
        "id": "sample_03",
        "source": "transcribed_voice",
        "text": (
            "Хочу поделиться впечатлениями от книги которую недавно прочитал. "
            "Это была книга про историю искусственного интеллекта. "
            "Автор подробно описывает как развивалась эта область с пятидесятых годов прошлого века. "
            "Очень интересно было читать про первые программы которые умели играть в шахматы. "
            "Потом про появление экспертных систем в восьмидесятых. "
            "И наконец про современные нейронные сети и машинное обучение. "
            "Книга написана простым языком без лишнего технического жаргона. "
            "Её можно рекомендовать всем кто интересуется историей технологий но не является специалистом. "
            "Единственный минус это то что книга немного устарела и не охватывает последние пять лет."
        ),
        "reference_summary": (
            "Книга об истории ИИ от 1950-х до наших дней: шахматные программы, экспертные системы, нейросети. "
            "Написана доступно, подойдёт неспециалистам. Минус — не охватывает последние пять лет."
        ),
    },
    {
        "id": "sample_04",
        "source": "transcribed_voice",
        "text": (
            "Today I want to talk about our team's progress on the machine learning project. "
            "We spent most of the week cleaning and preparing the dataset. "
            "The raw data had a lot of missing values and inconsistencies that needed to be fixed. "
            "After preprocessing we ended up with about eighty thousand clean samples. "
            "We tried three different models: logistic regression, random forest, and gradient boosting. "
            "The gradient boosting model performed best with an accuracy of ninety-two percent on validation. "
            "Next week we plan to tune the hyperparameters and test on the holdout set. "
            "We also need to write up the experiment documentation."
        ),
        "reference_summary": (
            "The team cleaned 80k samples from a messy dataset. "
            "Gradient boosting achieved 92% validation accuracy, outperforming logistic regression and random forest. "
            "Next steps: hyperparameter tuning and holdout evaluation."
        ),
    },
]


# ─────────────────────────────────────────────────────────────
# ROUGE метрики (реализация вручную + rouge-score если доступен)
# ─────────────────────────────────────────────────────────────

def tokenize_for_rouge(text: str) -> List[str]:
    """Простая токенизация для ROUGE."""
    import re
    return re.findall(r"\b\w+\b", text.lower())


def get_ngrams(tokens: List[str], n: int) -> Dict[Tuple, int]:
    """Получаем n-граммы с их количеством."""
    ngrams = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i : i + n])
        ngrams[gram] = ngrams.get(gram, 0) + 1
    return ngrams


def rouge_n(hypothesis: str, reference: str, n: int) -> Dict[str, float]:
    """Вычисляем ROUGE-N: precision, recall, F1."""
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
    """Длина наибольшей общей подпоследовательности (для ROUGE-L)."""
    m, n = len(x), len(y)
    # Оптимизация памяти: храним только две строки
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
    """Вычисляем ROUGE-L на основе LCS."""
    hyp_tokens = tokenize_for_rouge(hypothesis)
    ref_tokens = tokenize_for_rouge(reference)
    lcs = lcs_length(hyp_tokens, ref_tokens)
    precision = lcs / max(len(hyp_tokens), 1)
    recall = lcs / max(len(ref_tokens), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1}


# ─────────────────────────────────────────────────────────────
# Baseline методы
# ─────────────────────────────────────────────────────────────

def lead_n_summary(text: str, n: int) -> str:
    """Baseline Lead-N: берём первые N предложений."""
    sentences = split_sentences(preprocess_text(text))
    return " ".join(sentences[:n]) if sentences else text


def random_n_summary(text: str, n: int, seed: int = 42) -> str:
    """Baseline Random-N: случайные N предложений."""
    sentences = split_sentences(preprocess_text(text))
    if len(sentences) <= n:
        return text
    rng = random.Random(seed)
    chosen = sorted(rng.sample(range(len(sentences)), n))
    return " ".join(sentences[i] for i in chosen)


# ─────────────────────────────────────────────────────────────
# Оценка с разными гиперпараметрами (grid search)
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


def evaluate_config(
    config: ExperimentConfig,
    samples: List[Dict],
) -> ExperimentResult:
    """Оцениваем одну конфигурацию суммаризатора на всех тестовых примерах."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from summarizer import Summarizer

    summarizer = Summarizer(
        compression_ratio=config.compression_ratio,
        similarity_threshold=config.similarity_threshold,
        damping=config.damping,
    )
    # Переустанавливаем векторайзер с нужными ngram_range
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
        summary = summarizer.summarize(sample["text"])
        ref = sample["reference_summary"]

        r1 = rouge_n(summary, ref, 1)["f1"]
        r2 = rouge_n(summary, ref, 2)["f1"]
        rl = rouge_l(summary, ref)["f1"]

        src_words = len(sample["text"].split())
        summ_words = len(summary.split())
        comp = summ_words / max(src_words, 1)

        r1_scores.append(r1)
        r2_scores.append(r2)
        rl_scores.append(rl)
        comp_scores.append(comp)

    return ExperimentResult(
        config=config,
        avg_rouge1_f1=np.mean(r1_scores),
        avg_rouge2_f1=np.mean(r2_scores),
        avg_rougeL_f1=np.mean(rl_scores),
        avg_compression=np.mean(comp_scores),
    )


def run_grid_search(samples: List[Dict]) -> List[ExperimentResult]:
    """Перебираем гиперпараметры, ищем лучшую конфигурацию."""
    configs = [
        ExperimentConfig(cr, st, d, ngr)
        for cr in [0.25, 0.35, 0.45]
        for st in [0.03, 0.05, 0.10]
        for d in [0.80, 0.85, 0.90]
        for ngr in [(1, 1), (1, 2)]
    ]

    results = []
    for i, cfg in enumerate(configs):
        result = evaluate_config(cfg, samples)
        results.append(result)
        if (i + 1) % 10 == 0:
            logger.info(f"  Проверено конфигураций: {i+1}/{len(configs)}")

    return sorted(results, key=lambda r: r.avg_rouge1_f1, reverse=True)


# ─────────────────────────────────────────────────────────────
# Основной запуск
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Оценка суммаризатора TF-IDF + TextRank")
    print("=" * 65)
    print()

    summarizer = Summarizer()

    # ── 1. Базовая оценка нашей модели vs baseline ──
    print("── Сравнение с baseline методами ──\n")

    model_r1, model_rl = [], []
    lead_r1, lead_rl = [], []
    rand_r1, rand_rl = [], []

    for sample in TEST_SAMPLES:
        text = sample["text"]
        ref = sample["reference_summary"]

        # Наша модель
        our_summary = summarizer.summarize(text)
        n_target = len(split_sentences(our_summary))

        # Baselines (тот же размер выжимки)
        lead_summary = lead_n_summary(text, n_target)
        rand_summary = random_n_summary(text, n_target)

        # ROUGE
        r1_our = rouge_n(our_summary, ref, 1)["f1"]
        rl_our = rouge_l(our_summary, ref)["f1"]
        r1_lead = rouge_n(lead_summary, ref, 1)["f1"]
        rl_lead = rouge_l(lead_summary, ref)["f1"]
        r1_rand = rouge_n(rand_summary, ref, 1)["f1"]
        rl_rand = rouge_l(rand_summary, ref)["f1"]

        model_r1.append(r1_our)
        model_rl.append(rl_our)
        lead_r1.append(r1_lead)
        lead_rl.append(rl_lead)
        rand_r1.append(r1_rand)
        rand_rl.append(rl_rand)

        print(f"[{sample['id']}]")
        print(f"  Источник ({len(text.split())} слов): {text[:80]}...")
        print(f"  Наша модель:   ROUGE-1={r1_our:.3f}  ROUGE-L={rl_our:.3f}  → {our_summary[:80]}...")
        print(f"  Lead-{n_target}:        ROUGE-1={r1_lead:.3f}  ROUGE-L={rl_lead:.3f}")
        print(f"  Random-{n_target}:      ROUGE-1={r1_rand:.3f}  ROUGE-L={rl_rand:.3f}")
        print()

    print("── Средние метрики ──\n")
    print(f"  {'Метод':<20} {'ROUGE-1 F1':>12} {'ROUGE-L F1':>12}")
    print(f"  {'-'*44}")
    print(f"  {'TF-IDF+TextRank':<20} {np.mean(model_r1):>12.4f} {np.mean(model_rl):>12.4f}")
    print(f"  {'Lead-N':<20} {np.mean(lead_r1):>12.4f} {np.mean(lead_rl):>12.4f}")
    print(f"  {'Random-N':<20} {np.mean(rand_r1):>12.4f} {np.mean(rand_rl):>12.4f}")
    print()

    # ── 2. Grid search по гиперпараметрам ──
    print("── Grid Search по гиперпараметрам ──\n")
    logger.info("Запуск grid search (54 конфигурации)...")

    results = run_grid_search(TEST_SAMPLES)

    print("  Топ-5 конфигураций (по ROUGE-1 F1):\n")
    print(f"  {'#':<4} {'compression':>12} {'sim_threshold':>14} {'damping':>8} {'ngrams':>8} {'R1-F1':>8} {'R2-F1':>8} {'RL-F1':>8} {'compress%':>10}")
    print(f"  {'-'*86}")
    for rank, res in enumerate(results[:5], 1):
        c = res.config
        print(
            f"  {rank:<4} {c.compression_ratio:>12.2f} {c.similarity_threshold:>14.2f} "
            f"{c.damping:>8.2f} {str(c.ngram_range):>8} "
            f"{res.avg_rouge1_f1:>8.4f} {res.avg_rouge2_f1:>8.4f} {res.avg_rougeL_f1:>8.4f} "
            f"{res.avg_compression:>9.1%}"
        )

    print()
    best = results[0]
    print(f"  ✓ Лучшая конфигурация: compression_ratio={best.config.compression_ratio}, "
          f"similarity_threshold={best.config.similarity_threshold}, "
          f"damping={best.config.damping}, ngrams={best.config.ngram_range}")
    print(f"  ✓ ROUGE-1 F1 = {best.avg_rouge1_f1:.4f}")
    print()

    # ── 3. Сохраняем результаты ──
    log_path = Path(__file__).parent.parent / "logs" / "experiment_results.json"
    log_path.parent.mkdir(exist_ok=True)

    experiment_log = {
        "experiment": "textrank_grid_search",
        "n_samples": len(TEST_SAMPLES),
        "n_configs": 54,
        "best_config": {
            "compression_ratio": best.config.compression_ratio,
            "similarity_threshold": best.config.similarity_threshold,
            "damping": best.config.damping,
            "ngram_range": list(best.config.ngram_range),
        },
        "best_scores": {
            "rouge1_f1": round(best.avg_rouge1_f1, 4),
            "rouge2_f1": round(best.avg_rouge2_f1, 4),
            "rougeL_f1": round(best.avg_rougeL_f1, 4),
        },
        "baseline_lead_n": {
            "rouge1_f1": round(float(np.mean(lead_r1)), 4),
            "rougeL_f1": round(float(np.mean(lead_rl)), 4),
        },
        "baseline_random_n": {
            "rouge1_f1": round(float(np.mean(rand_r1)), 4),
            "rougeL_f1": round(float(np.mean(rand_rl)), 4),
        },
        "all_configs_top10": [
            {
                "rank": i + 1,
                "compression_ratio": r.config.compression_ratio,
                "similarity_threshold": r.config.similarity_threshold,
                "damping": r.config.damping,
                "ngram_range": list(r.config.ngram_range),
                "rouge1_f1": round(r.avg_rouge1_f1, 4),
                "rouge2_f1": round(r.avg_rouge2_f1, 4),
                "rougeL_f1": round(r.avg_rougeL_f1, 4),
                "compression": round(r.avg_compression, 3),
            }
            for i, r in enumerate(results[:10])
        ],
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(experiment_log, f, ensure_ascii=False, indent=2)

    print(f"  Результаты сохранены: {log_path}")
    print()
    print("=" * 65)
    print("  Эксперимент завершён.")
    print("=" * 65)


if __name__ == "__main__":
    main()