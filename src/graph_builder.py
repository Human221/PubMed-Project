"""Knowledge graph construction and visualization."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from nlp_pipeline import ProcessedArticle

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH_IMAGE = PROJECT_ROOT / "data" / "processed" / "knowledge_graph.png"


class KnowledgeGraphBuilder:
    """Build an entity graph from NLP-extracted entities and relations."""

    NODE_COLORS = {
        "DRUG": "#78C6A3",
        "DISEASE": "#FFB703",
        "EFFECT": "#8ECAE6",
        "SYMPTOM": "#FB8500",
        "DOSAGE": "#CDB4DB",
        "UNKNOWN": "#CCCCCC",
    }

    def build_graph(self, articles: list[ProcessedArticle]) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        for article in articles:
            entity_labels = {entity.text: entity.label for entity in article.entities}
            for entity in article.entities:
                graph.add_node(entity.text, label=entity.label, pmids={article.pmid})
            for relation in article.relations:
                for node in (relation.source, relation.target):
                    graph.add_node(node, label=entity_labels.get(node, "UNKNOWN"), pmids={article.pmid})
                graph.add_edge(
                    relation.source,
                    relation.target,
                    relation=relation.predicate,
                    sentence=relation.sentence,
                    pmid=article.pmid,
                )
        LOGGER.info("Built graph with %d nodes and %d edges", graph.number_of_nodes(), graph.number_of_edges())
        return graph

    def visualize(self, graph: nx.MultiDiGraph, output_file: Path = DEFAULT_GRAPH_IMAGE) -> None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if graph.number_of_nodes() == 0:
            LOGGER.warning("Graph is empty; saving placeholder visualization")
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "No entities extracted", ha="center", va="center", fontsize=16)
            plt.axis("off")
            plt.savefig(output_file, dpi=200, bbox_inches="tight")
            plt.close()
            return

        plt.figure(figsize=(14, 10))
        layout = nx.spring_layout(graph, seed=42, k=0.8)
        colors = [self.NODE_COLORS.get(graph.nodes[node].get("label", "UNKNOWN"), "#CCCCCC") for node in graph.nodes]
        nx.draw_networkx_nodes(graph, layout, node_color=colors, node_size=1800, alpha=0.9)
        nx.draw_networkx_labels(graph, layout, font_size=9, font_weight="bold")
        nx.draw_networkx_edges(graph, layout, arrows=True, arrowstyle="-|>", arrowsize=18, edge_color="#555555")
        edge_labels: dict[tuple[str, str], str] = {}
        for source, target, data in graph.edges(data=True):
            edge_labels[(source, target)] = data.get("relation", "associated_with")
        nx.draw_networkx_edge_labels(graph, layout, edge_labels=edge_labels, font_size=8)
        plt.title("Candesartan biomedical knowledge graph", fontsize=16)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_file, dpi=250, bbox_inches="tight")
        plt.close()
        LOGGER.info("Saved graph visualization to %s", output_file)
