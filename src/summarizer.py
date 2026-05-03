"""
Модуль суммаризации текста — классический ML без нейросетей.

Подход: TF-IDF + TextRank (графовый алгоритм ранжирования предложений)

Алгоритм:
1. Разбиваем текст на предложения (токенизация)
2. Для каждого предложения строим TF-IDF вектор (sklearn)
3. Считаем матрицу схожести между всеми парами предложений (cosine similarity)
4. Строим граф: узлы — предложения, рёбра — схожесть > порога
5. Запускаем PageRank на графе → получаем важность каждого предложения
6. Выбираем топ-N предложений, сохраняем порядок из оригинала

Это классическая задача unsupervised ML: никакой разметки, только математика.
"""

import re
import logging
import math
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Стоп-слова (русский + английский)
# ──────────────────────────────────────────────

STOPWORDS_RU = {
    "а", "без", "более", "бы", "был", "была", "были", "было", "быть", "в", "вам",
    "вас", "весь", "во", "вот", "все", "всего", "всех", "вы", "где", "да", "даже",
    "для", "до", "его", "её", "ей", "ему", "если", "есть", "ещё", "же", "за",
    "здесь", "и", "из", "или", "им", "их", "к", "как", "ко", "когда", "кто",
    "ли", "либо", "мне", "может", "мой", "мы", "на", "надо", "нас", "не", "него",
    "нет", "ни", "нибудь", "никогда", "ним", "них", "но", "ну", "о", "об",
    "однако", "он", "она", "они", "оно", "от", "очень", "по", "под", "при",
    "с", "со", "так", "также", "такой", "там", "те", "тем", "то", "того",
    "тоже", "той", "только", "том", "ты", "у", "уже", "хотя", "чего", "чем",
    "что", "чтобы", "чьё", "этим", "этого", "этой", "этом", "этот", "эту",
    "я", "вот", "вообще", "кстати", "ладно", "ну", "просто", "типа", "значит",
    "короче", "вроде", "именно", "прямо", "мол", "дескать", "итого", "итак",
}

STOPWORDS_EN = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "up", "about", "into", "through", "during", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may", "might", "shall",
    "can", "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "their", "this", "that", "these", "those", "so", "yet",
    "both", "either", "neither", "just", "very", "also", "not", "no", "than",
}

ALL_STOPWORDS = STOPWORDS_RU | STOPWORDS_EN


# ──────────────────────────────────────────────
# Предобработка текста
# ──────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """Нормализация текста: чистим лишние символы, пробелы."""
    # Убираем повторяющиеся пробелы
    text = re.sub(r"\s+", " ", text)
    # Убираем символы, кроме букв, цифр, знаков препинания
    text = re.sub(r"[^\w\s.,!?;:\-—–()\"\'«»\u0400-\u04FF]", " ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """
    Разбиваем текст на предложения.
    Учитываем особенности устной речи (отсутствие знаков препинания).
    """
    # Попытка разбить по знакам препинания
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # Если знаков нет (устная речь) — разбиваем по смысловым паузам
    # с помощью слов-маркеров и по длине
    if len(sentences) <= 1 or all(len(s.split()) < 3 for s in sentences):
        sentences = _split_by_discourse_markers(text)

    # Фильтруем слишком короткие «предложения» (< 4 слов)
    sentences = [s.strip() for s in sentences if len(s.split()) >= 4]

    return sentences


DISCOURSE_MARKERS = [
    r"\bпотому что\b", r"\bтак как\b", r"\bтаким образом\b",
    r"\bвo-первых\b", r"\bво-вторых\b", r"\bв-третьих\b",
    r"\bнапример\b", r"\bтем не менее\b", r"\bоднако\b",
    r"\bпоэтому\b", r"\bследовательно\b", r"\bкроме того\b",
    r"\bпри этом\b", r"\bвместе с тем\b", r"\bтогда как\b",
    r"\bbecause\b", r"\bhowever\b", r"\btherefore\b",
    r"\bfurthermore\b", r"\bmoreover\b", r"\bnevertheless\b",
    r"\bin addition\b", r"\bfor example\b", r"\bon the other hand\b",
]


def _split_by_discourse_markers(text: str) -> List[str]:
    """Разбиваем устную речь по маркерам дискурса и примерной длине."""
    # Ставим разделители перед маркерами
    for marker in DISCOURSE_MARKERS:
        text = re.sub(marker, r" |SPLIT| \g<0>", text, flags=re.IGNORECASE)

    parts = [p.strip() for p in text.split("|SPLIT|") if p.strip()]

    # Если части слишком длинные (> 60 слов) — дробим по ~30 слов
    result = []
    for part in parts:
        words = part.split()
        if len(words) > 60:
            for i in range(0, len(words), 30):
                chunk = " ".join(words[i : i + 30])
                if chunk:
                    result.append(chunk)
        else:
            result.append(part)

    return result if result else [text]


def tokenize_for_tfidf(text: str) -> str:
    """Токенизация для TF-IDF: нижний регистр, удаление стоп-слов."""
    words = re.findall(r"\b[а-яёa-z]{3,}\b", text.lower())
    filtered = [w for w in words if w not in ALL_STOPWORDS]
    return " ".join(filtered)


# ──────────────────────────────────────────────
# Основной класс Summarizer
# ──────────────────────────────────────────────

class Summarizer:
    """
    Extractive summarizer на основе TF-IDF + TextRank.

    Extractive суммаризация — выбираем самые важные предложения из оригинала.
    В отличие от abstractive (генеративной), мы не «придумываем» новый текст.
    
    Параметры:
        compression_ratio: Доля предложений для выжимки (0.0–1.0)
        min_sentences: Минимальное кол-во предложений в выжимке
        max_sentences: Максимальное кол-во предложений в выжимке
        similarity_threshold: Порог схожести для рёбер графа TextRank
        damping: Коэффициент затухания PageRank (стандарт: 0.85)
        max_iter: Макс. итераций PageRank
    """

    def __init__(
        self,
        compression_ratio: float = 0.35,
        min_sentences: int = 1,
        max_sentences: int = 5,
        similarity_threshold: float = 0.05,
        damping: float = 0.85,
        max_iter: int = 100,
        convergence_threshold: float = 1e-5,
    ):
        self.compression_ratio = compression_ratio
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.similarity_threshold = similarity_threshold
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_threshold = convergence_threshold

        # TF-IDF векторизатор
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            tokenizer=lambda t: t.split(),  # уже токенизировано
            lowercase=False,
            ngram_range=(1, 2),     # Унграммы + биграммы — учитываем контекст
            max_df=0.9,             # Игнорируем слова в > 90% предложений
            min_df=1,
            sublinear_tf=True,      # log(TF + 1) — сглаживаем частые слова
        )

    def summarize(self, text: str, n_sentences: int = None) -> str:
        """
        Основная функция суммаризации.

        Args:
            text: Входной текст
            n_sentences: Кол-во предложений (если None — вычисляется автоматически)

        Returns:
            Краткая выжимка текста
        """
        text = preprocess_text(text)
        sentences = split_sentences(text)

        logger.debug(f"Разбито на {len(sentences)} предложений")

        # Если текст слишком короткий — возвращаем как есть
        if len(sentences) <= 2:
            return text

        # Определяем целевое кол-во предложений
        if n_sentences is None:
            n_sentences = self._compute_target_sentences(len(sentences))

        # Получаем ранги предложений
        scores = self._textrank_scores(sentences)

        # Выбираем топ-N по рангу, сохраняем оригинальный порядок
        top_indices = sorted(
            sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:n_sentences]
        )

        summary_sentences = [sentences[i] for i in top_indices]
        summary = " ".join(summary_sentences)

        logger.debug(
            f"Суммаризация: {len(sentences)} → {n_sentences} предложений, "
            f"сжатие {len(summary.split())/max(len(text.split()),1):.1%}"
        )

        return summary

    def _compute_target_sentences(self, total: int) -> int:
        """Вычисляем оптимальное число предложений для выжимки."""
        target = max(
            self.min_sentences,
            min(
                self.max_sentences,
                round(total * self.compression_ratio),
            ),
        )
        return target

    def _textrank_scores(self, sentences: List[str]) -> np.ndarray:
        """
        Вычисляем TextRank-оценки для предложений.

        Шаги:
        1. TF-IDF векторизация предложений
        2. Матрица попарной схожести (cosine similarity)
        3. Граф → матрица перехода
        4. PageRank (power iteration)
        """
        n = len(sentences)

        # ── 1. TF-IDF векторы ──
        tokenized = [tokenize_for_tfidf(s) for s in sentences]

        # Если после токенизации нет слов — возвращаем равномерные веса
        if all(not t.strip() for t in tokenized):
            return np.ones(n) / n

        try:
            tfidf_matrix = self.vectorizer.fit_transform(tokenized)
        except ValueError:
            # Пустой словарь
            return np.ones(n) / n

        # ── 2. Матрица схожести ──
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Убираем самосхожесть и слабые связи
        np.fill_diagonal(similarity_matrix, 0.0)
        similarity_matrix[similarity_matrix < self.similarity_threshold] = 0.0

        # ── 3. Матрица перехода (нормализация по строкам) ──
        row_sums = similarity_matrix.sum(axis=1, keepdims=True)
        # Избегаем деления на ноль: изолированные узлы распределяют вес равномерно
        row_sums = np.where(row_sums == 0, 1e-9, row_sums)
        transition_matrix = similarity_matrix / row_sums

        # ── 4. PageRank (power iteration) ──
        scores = self._pagerank(transition_matrix)

        return scores

    def _pagerank(self, transition_matrix: np.ndarray) -> np.ndarray:
        """
        Итеративный PageRank.

        Формула: PR(i) = (1 - d)/N + d * Σ_j [ PR(j) * T(j,i) ]
        где d — damping factor, N — число узлов.
        """
        n = len(transition_matrix)
        # Начальное равномерное распределение
        scores = np.ones(n) / n
        teleport = (1.0 - self.damping) / n

        for iteration in range(self.max_iter):
            prev_scores = scores.copy()
            # Векторизованное обновление
            scores = teleport + self.damping * transition_matrix.T @ scores
            # Проверяем сходимость
            delta = np.abs(scores - prev_scores).sum()
            if delta < self.convergence_threshold:
                logger.debug(f"PageRank сошёлся за {iteration + 1} итераций")
                break

        return scores

    def get_sentence_scores(self, text: str) -> List[Tuple[str, float]]:
        """
        Возвращает предложения с их TextRank-оценками.
        Полезно для анализа и экспериментов.
        """
        text = preprocess_text(text)
        sentences = split_sentences(text)
        if not sentences:
            return []
        scores = self._textrank_scores(sentences)
        # Нормализуем оценки в диапазон [0, 1]
        max_score = scores.max() or 1.0
        pairs = [(s, float(sc / max_score)) for s, sc in zip(sentences, scores)]
        return sorted(pairs, key=lambda x: x[1], reverse=True)