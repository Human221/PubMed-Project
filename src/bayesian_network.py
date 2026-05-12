"""Bayesian network for Candesartan treatment scenarios."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

try:
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    try:
        from pgmpy.models import BayesianNetwork
    except ImportError:  # pgmpy >= 1.0 renamed the class
        from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError as exc:  # fail clearly when the optional dependency is absent
    raise ImportError("pgmpy is required for bayesian_network.py. Install requirements.txt first.") from exc

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class QueryResult:
    experiment: str
    probability: float
    interpretation: str
    evidence: dict[str, int]


class CandesartanBayesianNetwork:
    """Small explainable Bayesian network for classroom demonstrations."""

    def __init__(self) -> None:
        self.model = self._build_model()
        self.inference = VariableElimination(self.model)

    def _build_model(self) -> BayesianNetwork:
        model = BayesianNetwork(
            [
                ("Candesartan", "BloodPressureReduced"),
                ("Candesartan", "SideEffects"),
                ("BloodPressureReduced", "Hypertension"),
                ("Hypertension", "StrokeRisk"),
                ("Diabetes", "StrokeRisk"),
                ("LongTermTreatment", "BloodPressureReduced"),
                ("LongTermTreatment", "ConditionImproved"),
                ("BloodPressureReduced", "ConditionImproved"),
            ]
        )

        cpds = [
            TabularCPD("Candesartan", 2, [[0.5], [0.5]], state_names={"Candesartan": ["no", "yes"]}),
            TabularCPD("Diabetes", 2, [[0.75], [0.25]], state_names={"Diabetes": ["no", "yes"]}),
            TabularCPD("LongTermTreatment", 2, [[0.6], [0.4]], state_names={"LongTermTreatment": ["no", "yes"]}),
            TabularCPD(
                "BloodPressureReduced",
                2,
                [[0.85, 0.55, 0.35, 0.12], [0.15, 0.45, 0.65, 0.88]],
                evidence=["Candesartan", "LongTermTreatment"],
                evidence_card=[2, 2],
                state_names={
                    "BloodPressureReduced": ["no", "yes"],
                    "Candesartan": ["no", "yes"],
                    "LongTermTreatment": ["no", "yes"],
                },
            ),
            TabularCPD(
                "SideEffects",
                2,
                [[0.93, 0.82], [0.07, 0.18]],
                evidence=["Candesartan"],
                evidence_card=[2],
                state_names={"SideEffects": ["no", "yes"], "Candesartan": ["no", "yes"]},
            ),
            TabularCPD(
                "Hypertension",
                2,
                [[0.25, 0.80], [0.75, 0.20]],
                evidence=["BloodPressureReduced"],
                evidence_card=[2],
                state_names={"Hypertension": ["no", "yes"], "BloodPressureReduced": ["no", "yes"]},
            ),
            TabularCPD(
                "StrokeRisk",
                2,
                [[0.90, 0.78, 0.72, 0.45], [0.10, 0.22, 0.28, 0.55]],
                evidence=["Hypertension", "Diabetes"],
                evidence_card=[2, 2],
                state_names={"StrokeRisk": ["low", "high"], "Hypertension": ["no", "yes"], "Diabetes": ["no", "yes"]},
            ),
            TabularCPD(
                "ConditionImproved",
                2,
                [[0.75, 0.45, 0.35, 0.10], [0.25, 0.55, 0.65, 0.90]],
                evidence=["LongTermTreatment", "BloodPressureReduced"],
                evidence_card=[2, 2],
                state_names={
                    "ConditionImproved": ["no", "yes"],
                    "LongTermTreatment": ["no", "yes"],
                    "BloodPressureReduced": ["no", "yes"],
                },
            ),
        ]
        model.add_cpds(*cpds)
        if not model.check_model():
            raise ValueError("Invalid Bayesian network configuration")
        LOGGER.info("Bayesian network initialized with %d nodes", len(model.nodes()))
        return model

    def probability(self, variable: str, positive_state: str, evidence: dict[str, str] | None = None) -> float:
        """Return probability for a named variable state."""
        evidence = evidence or {}
        query = self.inference.query(variables=[variable], evidence=evidence, show_progress=False)
        states = query.state_names[variable]
        index = states.index(positive_state)
        return float(query.values[index])

    def describe_model(self) -> dict[str, Any]:
        return {"nodes": list(self.model.nodes()), "edges": list(self.model.edges())}
