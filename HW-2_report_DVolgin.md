# Отчёт по Домашнему заданию 2, ML System Design


**Студент:** Волгин Даниил Эдуардович

**Кейс:** "Классификация изображений котов и собак на породы"

**Тип данных:** Изображения

**Бизнес-цель:** Многоклассовая классификация (с похожими классами)

---


### **1. Эксперименты с моделями**

| Модель | Основная метрика (Macro F1) | Оптимизация гиперпараметров |
| --- | --- | --- |
| Resnet18 (Head train) | 0.8722 | Нет |
| Resnet18 (Full finetune) | 0.789 | Нет |
| Resnet18 (Head train -> Full finetune) | 0.90 | Нет |
| EfficientNetB3 (Head train -> Full finetune) | 0.86 | Нет |
| MobileNetLargev3 (Head train -> Full finetune) | 0.90 | Нет |
| ConvNextTiny (Head train -> Full finetune) | 0.95 | Нет |
| --- | --- | --- |
| MobileNetLargev3 (Head train -> Full finetune) | 0.90 | RandomSearch |
| Resnet18 (Head train -> Full finetune) | 0.89 | RandomSearch |

**Выбранная итоговая модель:** MobileNetLargev3

**Причина выбора:** Лучшее качество по основной метрике при малых требованиях к ресурсам.


---

### **2. Демо инференса**

✔ Тип интерфейса: Gradio

✔ Как запустить: Выполнить `python web.py`

✔ Воспроизводимость: Склонировать репозиторий, установить зависимости из prod-requirements.txt.

---

**Скриншоты метрик и интерфейса есть в ноутбуке "HW-2_DVolgin.ipynb"**
