"""Paper-defined 12-morphometric clinical narrative and JSON contract."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Morphometrics:
    langerhans_cell_density: float
    mean_field_area: float
    mean_dendrite_tips: float
    mean_cell_perimeter: float
    mean_cell_aspect_ratio: float
    mean_cell_solidity: float
    nerve_fiber_length: float
    nerve_fiber_density: float
    nerve_branch_point_density: float
    nerve_curvature_index: float
    nerve_fractal_dimension: float
    nerve_fiber_area: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "Morphometrics":
        expected = {field.name for field in fields(cls)}
        missing, extra = sorted(expected - values.keys()), sorted(values.keys() - expected)
        if missing or extra:
            raise ValueError(f"Invalid morphometrics keys; missing={missing}, extra={extra}.")
        converted = {name: float(values[name]) for name in expected}
        if not all(np.isfinite(value) for value in converted.values()):
            raise ValueError("All morphometric values must be finite numbers.")
        return cls(**converted)


FUNCTIONAL_GRADING_QUESTION = (
    "What are the most likely severity grades for corneal nerve tortuosity "
    "and Langerhans cell activation?"
)


def build_clinical_narrative(sample: Mapping[str, Any]) -> str:
    required = {"age", "gender", "systemic_comorbidities", "duration_years", "morphometrics"}
    missing = sorted(required - sample.keys())
    if missing:
        raise ValueError(f"Missing clinical fields: {missing}.")
    metrics = Morphometrics.from_mapping(sample["morphometrics"])
    comorbidities = sample["systemic_comorbidities"]
    if isinstance(comorbidities, str):
        comorbidity_text = comorbidities
    elif isinstance(comorbidities, Sequence):
        comorbidity_text = ", ".join(str(value) for value in comorbidities) or "no systemic comorbidities"
    else:
        raise ValueError("systemic_comorbidities must be a string or a list of strings.")
    context = (
        f"A {int(sample['age'])}-year-old {sample['gender']} with a history of {comorbidity_text} "
        f"for {float(sample['duration_years']):g} years. Quantitative analysis of a corneal confocal "
        "microscope image reveals the following morphometric parameters: "
        f"Langerhans cell density = {metrics.langerhans_cell_density:g} cells/mm2, "
        f"mean field area = {metrics.mean_field_area:g} um2, "
        f"mean dendrite tips per cell = {metrics.mean_dendrite_tips:g}, "
        f"mean cell perimeter = {metrics.mean_cell_perimeter:g} um, "
        f"mean cell aspect ratio = {metrics.mean_cell_aspect_ratio:g}, "
        f"mean cell solidity = {metrics.mean_cell_solidity:g}, "
        f"nerve fiber length = {metrics.nerve_fiber_length:g} mm/mm2, "
        f"nerve fiber density = {metrics.nerve_fiber_density:g} fibers/mm2, "
        f"nerve branch point density = {metrics.nerve_branch_point_density:g} points/mm2, "
        f"nerve curvature index = {metrics.nerve_curvature_index:g}, "
        f"nerve fractal dimension = {metrics.nerve_fractal_dimension:g}, "
        f"nerve fiber area = {metrics.nerve_fiber_area:g} mm2/mm2."
    )
    return f"{context} {FUNCTIONAL_GRADING_QUESTION}"


def load_text_samples(path: str | Path, require_labels: bool = True) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not payload:
        raise ValueError("The JSON root must be a non-empty list.")
    validated = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"Sample {index} must be an object.")
        sample = dict(raw)
        sample["input"] = sample.get("input") or build_clinical_narrative(sample)
        if require_labels:
            for name in ("nerve_label", "cell_label"):
                if name not in sample or isinstance(sample[name], bool) or int(sample[name]) not in range(4):
                    raise ValueError(f"Sample {index} requires {name} in {{0, 1, 2, 3}}.")
                sample[name] = int(sample[name])
        validated.append(sample)
    return validated
