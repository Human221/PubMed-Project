"""Command-line entry point for the PubMed Candesartan project."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from bayesian_network import CandesartanBayesianNetwork
from experiments import ExperimentRunner
from graph_builder import KnowledgeGraphBuilder
from nlp_pipeline import MedicalNLPPipeline
from pubmed_loader import DEFAULT_QUERY, PubMedLoader
from update_checker import UpdateChecker

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def ensure_directories() -> None:
    for relative_path in ["data/articles", "data/processed", "data/cache", "report_examples"]:
        (PROJECT_ROOT / relative_path).mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze PubMed articles about Candesartan")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="PubMed search query")
    parser.add_argument("--retmax", type=int, default=10, help="Number of PubMed articles to download")
    parser.add_argument("--email", default="student@example.com", help="Email required by NCBI Entrez")
    parser.add_argument("--api-key", default=None, help="Optional NCBI API key")
    parser.add_argument("--skip-download", action="store_true", help="Use cached JSON articles instead of downloading")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    ensure_directories()

    loader = PubMedLoader(email=args.email, api_key=args.api_key)
    articles_path = PROJECT_ROOT / "data" / "articles" / "candesartan_articles.json"

    if args.skip_download:
        articles = loader.load_articles(articles_path)
        if not articles:
            raise FileNotFoundError("No cached articles found. Run without --skip-download first.")
    else:
        articles = loader.download(query=args.query, retmax=args.retmax, output_file=articles_path)

    update_checker = UpdateChecker(loader)
    articles, update_report = update_checker.check_and_refresh(articles)
    loader.save_articles(articles, articles_path)
    logging.info("Update report: %s", update_report)

    nlp_pipeline = MedicalNLPPipeline()
    processed_articles = nlp_pipeline.process_articles(articles)

    graph_builder = KnowledgeGraphBuilder()
    graph = graph_builder.build_graph(processed_articles)
    graph_builder.visualize(graph)

    bayesian_network = CandesartanBayesianNetwork()
    logging.info("Bayesian model: %s", bayesian_network.describe_model())

    experiment_runner = ExperimentRunner(bayesian_network)
    experiment_runner.run_all()


if __name__ == "__main__":
    main()
