# PubMed Candesartan NLP/ML Project

Учебный Python-проект для анализа научных статей PubMed о препарате **Candesartan** с применением PubMed API, BioBERT, spaCy, графа знаний и байесовской сети.

## Архитектура проекта

```text
project/
├── data/
│   ├── articles/       # JSON со статьями PubMed
│   ├── processed/      # NLP-результаты, граф, CSV экспериментов
│   └── cache/          # MD5-хэши статей для проверки обновлений
├── src/
│   ├── pubmed_loader.py       # загрузка статей через BioPython Entrez
│   ├── update_checker.py      # проверка новых/изменённых статей
│   ├── nlp_pipeline.py        # BioBERT + spaCy + NER + связи
│   ├── graph_builder.py       # NetworkX-граф и визуализация
│   ├── bayesian_network.py    # pgmpy BayesianNetwork/CPD/inference
│   ├── experiments.py         # 5 вероятностных экспериментов
│   └── main.py                # полный запуск пайплайна
├── requirements.txt
├── README.md
└── report_examples/
```

## Установка зависимостей

Рекомендуется использовать виртуальное окружение Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> Если модель `en_core_web_sm` не установлена, проект автоматически переключится на `spacy.blank("en")` с sentencizer, но полноценный dependency parsing будет доступен только с установленной моделью spaCy.

## Запуск проекта

NCBI просит указывать email при обращении к Entrez API. Для учебного запуска:

```bash
python src/main.py --email your.email@example.com --retmax 10
```

Повторный запуск без новой загрузки статей:

```bash
python src/main.py --skip-download
```

Результаты сохраняются в:

- `data/articles/candesartan_articles.json` — статьи PubMed;
- `data/cache/hashes.json` — MD5-хэши статей;
- `data/processed/nlp_results.json` — сущности, зависимости и отношения;
- `data/processed/knowledge_graph.png` — визуализация графа знаний;
- `data/processed/experiment_results.csv` — результаты байесовских экспериментов.

## Как работает PubMed API

`src/pubmed_loader.py` использует официальный интерфейс NCBI через `Bio.Entrez`:

1. `Entrez.esearch` ищет PMID по запросу `Candesartan`.
2. `Entrez.efetch` получает XML-записи PubMed.
3. Парсер извлекает PMID, title, abstract, authors и publication year.
4. JSON сохраняется в `data/articles/`.

В код добавлены retry, задержка между запросами, обработка HTTP/API ошибок и логирование пустых abstract.

## Как работает BioBERT

В `src/nlp_pipeline.py` используется модель `dmis-lab/biobert-base-cased-v1.1` из Hugging Face Transformers. Это биомедицинский BERT, обученный на научных текстах. В проекте он применяется как transformer backbone для контекстных признаков текста, а требуемые учебные сущности (`DRUG`, `DISEASE`, `SYMPTOM`, `DOSAGE`, `EFFECT`) выделяются прозрачным rule-based NER-слоем.

Такой подход делает демонстрацию воспроизводимой: базовый BioBERT не содержит готовой token-classification головы для именно этих пяти классов, поэтому правила обеспечивают контролируемую разметку для отчёта.

## Как работает spaCy

spaCy используется для:

- разбиения текста на предложения;
- токенизации;
- dependency parsing при установленной модели `en_core_web_sm`;
- построения связей вида `(Candesartan) --reduces--> (blood pressure)`.

Пример:

```text
Candesartan reduces blood pressure
```

Будет преобразован в отношение:

```text
Candesartan --reduces--> blood pressure
```

## Как работает граф знаний

`src/graph_builder.py` строит `networkx.MultiDiGraph`:

- узлы: лекарства, заболевания, эффекты, симптомы и дозировки;
- рёбра: `treats`, `causes`, `reduces`, `associated_with`;
- визуализация сохраняется как `data/processed/knowledge_graph.png`.

Цвет узла зависит от типа сущности, подписи рёбер показывают тип отношения.

## Как работает байесовская сеть

`src/bayesian_network.py` создаёт вероятностную модель на `pgmpy` со структурой, включающей примерную цепочку:

```text
Candesartan -> BloodPressureReduced -> Hypertension -> StrokeRisk
```

Также учитываются `Diabetes`, `SideEffects`, `LongTermTreatment` и `ConditionImproved`. Для узлов заданы `TabularCPD`, а запросы выполняются через `VariableElimination`.

## Эксперименты

`src/experiments.py` запускает 5 экспериментов:

1. Вероятность снижения давления при приёме Candesartan.
2. Вероятность инсульта без лечения.
3. Вероятность побочных эффектов.
4. Вероятность осложнений при диабете.
5. Вероятность улучшения состояния при длительном лечении.

Каждый эксперимент выводит входные данные, вероятность и интерпретацию, а затем сохраняет CSV-файл для отчёта.

## Примечания для демонстрации

- Для настоящей загрузки PubMed нужен интернет.
- Первый запуск BioBERT скачивает веса модели Hugging Face и может занять несколько минут.
- Вероятности в байесовской сети являются демонстрационными и не предназначены для медицинских решений.
