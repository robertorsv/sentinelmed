from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

LABELS = ["CURP", "NSS", "RFC", "EMAIL", "PHONE", "BIRTHDATE", "ADDRESS", "NAME", "CUSTOM"]
# When two candidates overlap, the more specific label wins.
PRIORITY = {label: i for i, label in enumerate(LABELS)}


@dataclass
class Finding:
    start: int
    end: int
    text: str
    label: str
    sources: set = field(default_factory=set)   # rule | context | gazetteer | ner | propagated | column | custom
    prior: float = 0.5                           # the detector's own guess, before the classifier
    valid: Optional[bool] = None                 # checksum result where one exists (CURP, NSS)
    role: Optional[str] = None                   # NAME only: patient | relative | doctor | person
    confidence: float = 0.0
    decision: str = "mask"                       # mask | review | skip-review | skip (see Confirmer)
    features: dict = field(default_factory=dict)
    placeholder: str = ""
    reason: str = ""                             # why it was left unmasked, when it was

    @property
    def masked(self) -> bool:
        return self.decision in ("mask", "review")
