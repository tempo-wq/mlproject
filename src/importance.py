import logging
from transformers import pipeline

logger = logging.getLogger(__name__)

class ImportanceScorer:
    def __init__(self):
        logger.info("Загрузка модели Zero-Shot Classification (ruBERT-tiny)...")
        # Используем крошечную модель, идеальную для хакатонов и русского языка
        self.classifier = pipeline(
            "zero-shot-classification", 
            model="cointegrated/rubert-tiny-bilingual-nli"
        )
        # Классы, по которым мы будем оценивать текст
        self.candidate_labels = ["срочная задача", "важная информация", "просто болтовня", "эмоции"]

    def score(self, text: str) -> dict:
        """
        Возвращает словарь с вероятностями.
        Чем ближе 'срочная задача' или 'важная информация' к 1.0, тем важнее текст.
        """
        if not text.strip():
            return {"score": 0.0, "is_important": False}

        result = self.classifier(text, self.candidate_labels, multi_label=True)
        
        # Собираем скоры в удобный словарь
        scores = dict(zip(result['labels'], result['scores']))
        
        # Считаем совокупный "скор важности" (сумма вероятностей срочности и важности)
        urgency_score = scores.get("срочная задача", 0)
        info_score = scores.get("важная информация", 0)
        
        total_importance = (urgency_score * 0.6) + (info_score * 0.4) # Срочность имеет больший вес
        
        return {
            "score": round(total_importance, 2),
            "is_important": total_importance > 0.5, # Порог подбирается эмпирически
            "raw_scores": scores
        }

# Для теста:
#if __name__ == "__main__":
#    scorer = ImportanceScorer()
#     print(scorer.score("Завтра в 10 утра дедлайн по проекту, обязательно пришли презентацию!"))