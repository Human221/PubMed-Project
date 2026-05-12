"""Article update detection based on content hashes."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pubmed_loader import PubMedArticle, PubMedLoader

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HASH_FILE = PROJECT_ROOT / "data" / "cache" / "hashes.json"


class UpdateChecker:
    """Track article content changes and refresh changed PubMed records."""

    def __init__(self, loader: PubMedLoader, hash_file: Path = DEFAULT_HASH_FILE) -> None:
        self.loader = loader
        self.hash_file = hash_file
        self.hash_file.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def article_hash(article: PubMedArticle) -> str:
        """Compute a stable MD5 hash for meaningful article fields."""
        payload = json.dumps(asdict(article), sort_keys=True, ensure_ascii=False)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def load_hashes(self) -> dict[str, str]:
        if not self.hash_file.exists():
            return {}
        with self.hash_file.open("r", encoding="utf-8") as file:
            return json.load(file)

    def save_hashes(self, hashes: dict[str, str]) -> None:
        with self.hash_file.open("w", encoding="utf-8") as file:
            json.dump(hashes, file, ensure_ascii=False, indent=2)

    def check_and_refresh(self, articles: list[PubMedArticle]) -> tuple[list[PubMedArticle], dict[str, Any]]:
        """Detect new/changed articles and re-download changed records.

        PubMed records do not expose a classic version number through the simple
        metadata subset used here. Therefore a changed MD5 hash is treated as a
        new content version.
        """
        old_hashes = self.load_hashes()
        new_hashes: dict[str, str] = {}
        refreshed_articles: list[PubMedArticle] = []
        report: dict[str, Any] = {"new": [], "changed": [], "unchanged": [], "failed_refresh": []}

        for article in articles:
            current_hash = self.article_hash(article)
            old_hash = old_hashes.get(article.pmid)
            if old_hash is None:
                report["new"].append(article.pmid)
                refreshed_articles.append(article)
                new_hashes[article.pmid] = current_hash
            elif old_hash != current_hash:
                report["changed"].append(article.pmid)
                latest = self.loader.fetch_article(article.pmid)
                if latest is None:
                    LOGGER.warning("Could not refresh changed PMID %s; keeping local copy", article.pmid)
                    report["failed_refresh"].append(article.pmid)
                    latest = article
                refreshed_articles.append(latest)
                new_hashes[latest.pmid] = self.article_hash(latest)
            else:
                report["unchanged"].append(article.pmid)
                refreshed_articles.append(article)
                new_hashes[article.pmid] = current_hash

        self.save_hashes(new_hashes)
        LOGGER.info("Update check report: %s", report)
        return refreshed_articles, report
