
"""
ربط نماذج Hugging Face - مشروع NLP
"""

from transformers import pipeline
import pandas as pd
from tqdm import tqdm

class HuggingFaceConnector:
    """
    فئة للربط مع نماذج Hugging Face
    """

    def __init__(self, task, model_name, device=-1):
        self.task = task
        self.model_name = model_name
        self.device = device
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self):
        return pipeline(
            self.task,
            model=self.model_name,
            device=self.device
        )

    def predict(self, text):
        return self.pipeline(text)[0]

    def predict_batch(self, texts, batch_size=16):
        results = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            batch_results = self.pipeline(batch)
            results.extend(batch_results)
        return results


if __name__ == "__main__":
    # مثال الاستخدام
    connector = HuggingFaceConnector(
        task="sentiment-analysis",
        model_name="CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"
    )

    result = connector.predict("الخدمة ممتازة جداً")
    print(result)
