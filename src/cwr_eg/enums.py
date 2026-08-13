from __future__ import annotations

from enum import Enum


class DecisionLabel(str, Enum):
    KNOWN_SCHEME_KNOWN_KEY = "known_scheme_known_key"
    KNOWN_SCHEME_UNKNOWN_KEY = "known_scheme_unknown_key"
    SUSPECTED_UNKNOWN_SCHEME = "suspected_unknown_scheme"
    UNCERTAIN = "uncertain"
    NONE = "none"


class ValidityStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class KeyStatus(str, Enum):
    REGISTERED = "registered"
    SCHEME_ONLY = "scheme_only"
    NOT_APPLICABLE = "not_applicable"


class Applicability(str, Enum):
    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"


class TailDirection(str, Enum):
    UPPER = "upper"
    LOWER = "lower"
    TWO_SIDED = "two_sided"
