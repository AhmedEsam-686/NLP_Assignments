
# تقرير مقارنة ادوات معالجة النصوص العربية

## 1. Tokenization
- Spacy: متوسطة السرعة، دقة عالية
- NLTK: سريعة، دقة جيدة
- Simple: سريعة جدا، دقة منخفضة

الافضل: Spacy للمهام المتقدمة

## 2. Stemming
- ISRI: سريعة، دقة جيدة للعربية

الافضل: ISRI Stemmer

## 3. Lemmatization
- Qalsadi: بطيئة، دقة عالية جدا
- Simple Rules: سريعة، دقة منخفضة

الافضل: Qalsadi للدقة العالية

## التوصيات
1. Tokenization: Spacy
2. Stemming: ISRI
3. Lemmatization: Qalsadi
4. التنظيف: pyarabic + regex
