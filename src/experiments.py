"""Experiment runner for the Candesartan Bayesian network."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from bayesian_network import CandesartanBayesianNetwork

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_FILE = PROJECT_ROOT / "data" / "processed" / "experiment_results.csv"


@dataclass(slots=True)
class ExperimentResult:
    experiment: str
    input_data: str
    probability: float
    interpretation: str


class ExperimentRunner:
    """Runs required classroom experiments and exports results to CSV."""

    def __init__(self, network: CandesartanBayesianNetwork, output_file: Path = DEFAULT_RESULTS_FILE) -> None:
        self.network = network
        self.output_file = output_file
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def run_all(self) -> list[ExperimentResult]:
        experiments = [
            (
                "Вероятность снижения давления при приёме Candesartan",
                "Candesartan=yes",
                self.network.probability("BloodPressureReduced", "yes", {"Candesartan": "yes"}),
                "Модель ожидает клинически значимое снижение давления при применении препарата.",
            ),
            (
                "Вероятность инсульта без лечения",
                "Candesartan=no, Diabetes=no",
                self.network.probability("StrokeRisk", "high", {"Candesartan": "no", "Diabetes": "no"}),
                "Отсутствие лечения повышает вероятность сохранения гипертензии и связанного риска инсульта.",
            ),
            (
                "Вероятность побочных эффектов",
                "Candesartan=yes",
                self.network.probability("SideEffects", "yes", {"Candesartan": "yes"}),
                "Побочные эффекты возможны, но их вероятность ниже ожидаемой пользы снижения давления.",
            ),
            (
                "Вероятность осложнений при диабете",
                "Diabetes=yes, Hypertension=yes",
                self.network.probability("StrokeRisk", "high", {"Diabetes": "yes", "Hypertension": "yes"}),
                "Комбинация диабета и гипертензии формирует наиболее неблагоприятный профиль риска.",
            ),
            (
                "Вероятность улучшения состояния при длительном лечении",
                "Candesartan=yes, LongTermTreatment=yes",
                self.network.probability(
                    "ConditionImproved",
                    "yes",
                    {"Candesartan": "yes", "LongTermTreatment": "yes"},
                ),
                "Длительное лечение повышает шансы стабилизации давления и общего улучшения состояния.",
            ),
        ]
        results = [ExperimentResult(name, inputs, probability, interpretation) for name, inputs, probability, interpretation in experiments]
        self.save(results)
        for result in results:
            LOGGER.info("%s | %s | p=%.3f | %s", result.experiment, result.input_data, result.probability, result.interpretation)
        return results

    def save(self, results: list[ExperimentResult]) -> None:
        frame = pd.DataFrame([asdict(result) for result in results])
        frame.to_csv(self.output_file, index=False, encoding="utf-8")
        LOGGER.info("Saved experiment results to %s", self.output_file)
