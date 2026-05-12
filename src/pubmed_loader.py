"""PubMed article downloader for Candesartan literature.

The module uses the official NCBI Entrez/PubMed API through BioPython.
It stores a normalized JSON representation of articles in ``data/articles``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional
from urllib.error import HTTPError, URLError

from Bio import Entrez
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTICLES_DIR = PROJECT_ROOT / "data" / "articles"
DEFAULT_EMAIL = "student@example.com"
DEFAULT_QUERY = "Candesartan"


@dataclass(slots=True)
class PubMedArticle:
    """Normalized subset of PubMed metadata used by the project."""

    pmid: str
    title: str
    abstract: str
    authors: list[str]
    publication_year: Optional[int]


class PubMedLoader:
    """Download and serialize PubMed articles via BioPython Entrez."""

    def __init__(
        self,
        email: str = DEFAULT_EMAIL,
        api_key: Optional[str] = None,
        articles_dir: Path = DEFAULT_ARTICLES_DIR,
        request_delay_seconds: float = 0.34,
        max_retries: int = 3,
    ) -> None:
        self.email = email
        self.api_key = api_key
        self.articles_dir = articles_dir
        self.request_delay_seconds = request_delay_seconds
        self.max_retries = max_retries
        self.articles_dir.mkdir(parents=True, exist_ok=True)

        Entrez.email = self.email
        if self.api_key:
            Entrez.api_key = self.api_key

    def search(self, query: str = DEFAULT_QUERY, retmax: int = 10) -> list[str]:
        """Return PubMed identifiers for the query."""
        LOGGER.info("Searching PubMed: query=%r retmax=%s", query, retmax)
        try:
            with Entrez.esearch(db="pubmed", term=query, retmax=retmax, sort="relevance") as handle:
                record = Entrez.read(handle)
        except (HTTPError, URLError, RuntimeError) as exc:
            LOGGER.exception("PubMed search failed: %s", exc)
            raise

        pmids = [str(pmid) for pmid in record.get("IdList", [])]
        LOGGER.info("Found %d PubMed ids", len(pmids))
        return pmids

    def fetch_article(self, pmid: str) -> Optional[PubMedArticle]:
        """Fetch one PubMed article with retry and rate-limit handling."""
        for attempt in range(1, self.max_retries + 1):
            try:
                LOGGER.debug("Fetching PMID %s (attempt %d)", pmid, attempt)
                with Entrez.efetch(db="pubmed", id=pmid, rettype="xml", retmode="xml") as handle:
                    records = Entrez.read(handle)
                time.sleep(self.request_delay_seconds)
                return self._parse_article(records)
            except HTTPError as exc:
                LOGGER.warning("HTTP error for PMID %s: %s", pmid, exc)
                if exc.code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    time.sleep(self.request_delay_seconds * attempt * 3)
                    continue
                raise
            except (URLError, RuntimeError) as exc:
                LOGGER.warning("Temporary PubMed error for PMID %s: %s", pmid, exc)
                if attempt < self.max_retries:
                    time.sleep(self.request_delay_seconds * attempt * 2)
                    continue
                raise
        return None

    def fetch_articles(self, pmids: Iterable[str]) -> list[PubMedArticle]:
        """Fetch several articles and skip records with unrecoverable errors."""
        articles: list[PubMedArticle] = []
        for pmid in tqdm(list(pmids), desc="Downloading PubMed articles"):
            try:
                article = self.fetch_article(pmid)
            except Exception as exc:  # keep batch download resilient for demonstrations
                LOGGER.error("Skipping PMID %s after fetch failure: %s", pmid, exc)
                continue
            if article is None:
                LOGGER.warning("Skipping PMID %s: no parsable article returned", pmid)
                continue
            if not article.abstract.strip():
                LOGGER.warning("PMID %s has an empty abstract; article is kept with empty text", pmid)
            articles.append(article)
        return articles

    def download(
        self,
        query: str = DEFAULT_QUERY,
        retmax: int = 10,
        output_file: Path | None = None,
    ) -> list[PubMedArticle]:
        """Search, fetch, and persist articles for a query."""
        output_file = output_file or self.articles_dir / "candesartan_articles.json"
        pmids = self.search(query=query, retmax=retmax)
        articles = self.fetch_articles(pmids)
        self.save_articles(articles, output_file)
        return articles

    def save_articles(self, articles: list[PubMedArticle], output_file: Path) -> None:
        """Save articles as UTF-8 JSON."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as file:
            json.dump([asdict(article) for article in articles], file, ensure_ascii=False, indent=2)
        LOGGER.info("Saved %d articles to %s", len(articles), output_file)

    def load_articles(self, input_file: Path | None = None) -> list[PubMedArticle]:
        """Load previously downloaded articles from JSON."""
        input_file = input_file or self.articles_dir / "candesartan_articles.json"
        if not input_file.exists():
            return []
        with input_file.open("r", encoding="utf-8") as file:
            raw_articles = json.load(file)
        return [PubMedArticle(**article) for article in raw_articles]

    @staticmethod
    def _parse_article(records: Any) -> Optional[PubMedArticle]:
        """Parse BioPython's PubMed XML object into ``PubMedArticle``."""
        articles = records.get("PubmedArticle", [])
        if not articles:
            return None

        medline = articles[0].get("MedlineCitation", {})
        article = medline.get("Article", {})
        pmid = str(medline.get("PMID", ""))
        title = str(article.get("ArticleTitle", "")).strip()

        abstract_parts = article.get("Abstract", {}).get("AbstractText", [])
        abstract = " ".join(str(part) for part in abstract_parts).strip()

        authors = PubMedLoader._parse_authors(article.get("AuthorList", []))
        publication_year = PubMedLoader._parse_publication_year(article)

        return PubMedArticle(
            pmid=pmid,
            title=title,
            abstract=abstract,
            authors=authors,
            publication_year=publication_year,
        )

    @staticmethod
    def _parse_authors(author_list: Iterable[Any]) -> list[str]:
        authors: list[str] = []
        for author in author_list:
            last_name = str(author.get("LastName", "")).strip()
            fore_name = str(author.get("ForeName", "")).strip()
            collective = str(author.get("CollectiveName", "")).strip()
            if collective:
                authors.append(collective)
            elif last_name or fore_name:
                authors.append(f"{fore_name} {last_name}".strip())
        return authors

    @staticmethod
    def _parse_publication_year(article: dict[str, Any]) -> Optional[int]:
        journal_issue = article.get("Journal", {}).get("JournalIssue", {})
        pub_date = journal_issue.get("PubDate", {})
        article_dates = article.get("ArticleDate", []) or [{}]
        candidates = [pub_date.get("Year"), article_dates[0].get("Year")]
        for candidate in candidates:
            if candidate:
                try:
                    return int(str(candidate)[:4])
                except ValueError:
                    continue
        return None
