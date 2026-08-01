"""Vision branch: topology-aware pre-training followed by grading adaptation."""

from .grading_adaptation.model import DualTaskVisionClassifier
from .topology_aware_pretraining.model import TopologyAutoencoder

__all__ = ["DualTaskVisionClassifier", "TopologyAutoencoder"]
