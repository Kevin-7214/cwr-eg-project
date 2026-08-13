"""CWR-EG finite-open-set watermark localization and attribution."""

from cwr_eg.contracts import (
    CharacterInterval,
    DocumentDecision,
    GenericResidualEvidence,
    RegisteredEvidence,
    SegmentDecision,
    ValidityDiagnostics,
)
from cwr_eg.enums import DecisionLabel

__all__ = [
    "CharacterInterval",
    "DecisionLabel",
    "DocumentDecision",
    "GenericResidualEvidence",
    "RegisteredEvidence",
    "SegmentDecision",
    "ValidityDiagnostics",
]

__version__ = "0.1.0"
