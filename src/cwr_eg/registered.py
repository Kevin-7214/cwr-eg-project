from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

from cwr_eg.contracts import CharacterInterval, RegisteredEvidence
from cwr_eg.enums import TailDirection
from cwr_eg.hashing import sha256_text


@dataclass(frozen=True, slots=True)
class DetectorDeclaration:
    detector_id: str
    scheme_id: str
    scheme_family: str
    tokenizer_id: str
    tokenizer_revision: str
    requires_key: bool
    supports_scheme_only: bool
    minimum_effective_length: int
    supported_languages: tuple[str, ...]
    tail_direction: TailDirection
    emits_local_evidence: bool
    license: str
    source_url: str
    source_revision: str


@dataclass(frozen=True, slots=True)
class AuthorizedKey:
    key_id: str
    secret: Any

    @property
    def key_id_hash(self) -> str:
        return "sha256:" + sha256_text(self.key_id)


class RegisteredDetectorAdapter(ABC):
    declaration: DetectorDeclaration

    @abstractmethod
    def score(
        self,
        raw_text: str,
        interval: CharacterInterval,
        language: str,
        authorized_key: AuthorizedKey | None,
        context: Mapping[str, Any],
    ) -> RegisteredEvidence:
        """Return raw registered evidence without assigning a five-class label."""

    def validate_request(
        self, language: str, authorized_key: AuthorizedKey | None, effective_length: int
    ) -> None:
        declaration = self.declaration
        if language not in declaration.supported_languages:
            raise ValueError("unsupported_language")
        if effective_length < declaration.minimum_effective_length:
            raise ValueError("insufficient_effective_length")
        if declaration.requires_key and authorized_key is None:
            raise PermissionError("registered_detector_requires_authorized_key")
