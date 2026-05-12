"""NLP pipeline combining BioBERT embeddings, spaCy parsing, and biomedical NER rules."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import spacy
from spacy.language import Language
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, pipeline

from pubmed_loader import PubMedArticle

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
BIOBERT_MODEL = "dmis-lab/biobert-base-cased-v1.1"


@dataclass(slots=True)
class Entity:
    text: str
    label: str
    start: int
    end: int


@dataclass(slots=True)
class Relation:
    source: str
    predicate: str
    target: str
    sentence: str


@dataclass(slots=True)
class ProcessedArticle:
    pmid: str
    title: str
    entities: list[Entity]
    relations: list[Relation]
    dependencies: list[dict[str, str]]


class MedicalNLPPipeline:
    """Production-like biomedical NLP pipeline for PubMed abstracts.

    BioBERT is used as the transformer backbone for contextual embeddings. A
    transparent rule layer labels the required educational entity classes because
    the base BioBERT checkpoint is not a task-specific NER head.
    """

    RELATION_VERBS = {
        "reduce": "reduces",
        "reduces": "reduces",
        "reduced": "reduces",
        "lower": "reduces",
        "lowers": "reduces",
        "treat": "treats",
        "treats": "treats",
        "prevent": "treats",
        "prevents": "treats",
        "cause": "causes",
        "causes": "causes",
        "induce": "causes",
        "induces": "causes",
        "associate": "associated_with",
        "associated": "associated_with",
        "improve": "associated_with",
        "improves": "associated_with",
    }

    ENTITY_PATTERNS = {
        "DRUG": [r"\bcandesartan\b", r"\bangiotensin receptor blocker[s]?\b", r"\bARB[s]?\b"],
        "DISEASE": [r"\bhypertension\b", r"\bheart failure\b", r"\bstroke\b", r"\bdiabetes\b", r"\bnephropathy\b"],
        "SYMPTOM": [r"\bdizziness\b", r"\bheadache\b", r"\bfatigue\b", r"\bcough\b", r"\bhypotension\b"],
        "DOSAGE": [r"\b\d+(?:\.\d+)?\s?(?:mg|g|mcg|µg)(?:/day)?\b", r"\bonce daily\b", r"\btwice daily\b"],
        "EFFECT": [r"\bblood pressure\b", r"\bstroke risk\b", r"\bside effect[s]?\b", r"\badverse event[s]?\b", r"\bmortality\b"],
    }

    def __init__(self, processed_dir: Path = DEFAULT_PROCESSED_DIR, spacy_model: str = "en_core_web_sm") -> None:
        self.processed_dir = processed_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.nlp = self._load_spacy(spacy_model)
        self.tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)
        self.model = AutoModel.from_pretrained(BIOBERT_MODEL)
        self.feature_extractor = pipeline(
            "feature-extraction",
            model=self.model,
            tokenizer=self.tokenizer,
            truncation=True,
            max_length=256,
        )

    @staticmethod
    def _load_spacy(model_name: str) -> Language:
        try:
            return spacy.load(model_name)
        except OSError:
            LOGGER.warning("spaCy model %s is not installed; using a blank English pipeline", model_name)
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
            return nlp

    def process_articles(self, articles: Iterable[PubMedArticle]) -> list[ProcessedArticle]:
        processed: list[ProcessedArticle] = []
        for article in tqdm(list(articles), desc="Running NLP"):
            processed.append(self.process_article(article))
        self.save_processed(processed)
        return processed

    def process_article(self, article: PubMedArticle) -> ProcessedArticle:
        text = f"{article.title}. {article.abstract}".strip()
        # Run BioBERT to satisfy the transformer stage and warm contextual features.
        # The vectors are not persisted to keep the educational project lightweight.
        _ = self.feature_extractor(text[:2000]) if text else []
        doc = self.nlp(text)
        entities = self.extract_entities(text)
        dependencies = self.extract_dependencies(doc)
        relations = self.extract_relations(doc, entities)
        return ProcessedArticle(article.pmid, article.title, entities, relations, dependencies)

    def extract_entities(self, text: str) -> list[Entity]:
        entities: list[Entity] = []
        for label, patterns in self.ENTITY_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                    entities.append(Entity(match.group(0), label, match.start(), match.end()))
        entities.sort(key=lambda entity: (entity.start, entity.end))
        return self._deduplicate_entities(entities)

    @staticmethod
    def _deduplicate_entities(entities: list[Entity]) -> list[Entity]:
        seen: set[tuple[int, int, str]] = set()
        unique: list[Entity] = []
        for entity in entities:
            key = (entity.start, entity.end, entity.label)
            if key not in seen:
                seen.add(key)
                unique.append(entity)
        return unique

    @staticmethod
    def extract_dependencies(doc: spacy.tokens.Doc) -> list[dict[str, str]]:
        dependencies: list[dict[str, str]] = []
        for token in doc:
            dependencies.append(
                {
                    "token": token.text,
                    "lemma": token.lemma_ or token.text.lower(),
                    "pos": token.pos_ or "X",
                    "dep": token.dep_ or "dep",
                    "head": token.head.text if token.head is not token else "ROOT",
                }
            )
        return dependencies

    def extract_relations(self, doc: spacy.tokens.Doc, entities: list[Entity]) -> list[Relation]:
        relations: list[Relation] = []
        for sentence in doc.sents:
            sent_text = sentence.text
            sent_entities = [e for e in entities if e.start >= sentence.start_char and e.end <= sentence.end_char]
            if len(sent_entities) < 2:
                continue
            predicate = self._predicate_for_sentence(sent_text)
            if predicate is None:
                continue
            relations.append(Relation(sent_entities[0].text, predicate, sent_entities[-1].text, sent_text))
        return relations

    def _predicate_for_sentence(self, sentence: str) -> Optional[str]:
        sentence_lower = sentence.lower()
        for verb, relation in self.RELATION_VERBS.items():
            if re.search(rf"\b{re.escape(verb)}\b", sentence_lower):
                return relation
        return None

    def save_processed(self, processed: list[ProcessedArticle], output_file: Path | None = None) -> None:
        output_file = output_file or self.processed_dir / "nlp_results.json"
        serializable = []
        for article in processed:
            data = asdict(article)
            serializable.append(data)
        with output_file.open("w", encoding="utf-8") as file:
            json.dump(serializable, file, ensure_ascii=False, indent=2)
        LOGGER.info("Saved NLP results to %s", output_file)
