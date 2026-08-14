"""Repository-owned deterministic synthetic graph truth."""

from junctionlens.synthetic.corpus import (
    SyntheticCorpus,
    SyntheticCorpusError,
    generate_corpus,
    verify_corpus,
    write_corpus,
)
from junctionlens.synthetic.generator import (
    GeneratedCorruption,
    GeneratedSceneFrame,
    generate_corruptions,
    generate_scene_frames,
)
from junctionlens.synthetic.models import CorruptionKind, SceneKind

__all__ = [
    "CorruptionKind",
    "GeneratedCorruption",
    "GeneratedSceneFrame",
    "SceneKind",
    "SyntheticCorpus",
    "SyntheticCorpusError",
    "generate_corpus",
    "generate_corruptions",
    "generate_scene_frames",
    "verify_corpus",
    "write_corpus",
]
