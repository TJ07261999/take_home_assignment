"""Reznar's Arcane Oddities - magic item ontology.

The catalog PDF is image-based, so the extraction pipeline keeps two layers:

1. Source facts: what the catalog explicitly says on the page, such as the
   item name, printed type line, rarity, attunement, and concise mechanics
   summary. Full page OCR text is stored on ExtractionPage for auditability.
2. Semantic tags: queryable interpretation used for pattern finding, such as
   equipment slots, target creature types, damage types, spells, and limits.

This separation lets us preserve page-level provenance while still giving
Reznar searchable fields for recommendations and later rarity analysis.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator


# ---- Markers + validators ---------------------------------------------------


class Hint:
    """Free-text description for LLM prompts, carried as Annotated metadata."""

    __slots__ = ("text",)

    def __init__(self, text: str):
        self.text = text


def _clean_text(v: Any) -> str:
    if v is None:
        return ""
    return re.sub(r"\s+", " ", str(v)).strip()


def _clean_optional_text(v: Any) -> str | None:
    text = _clean_text(v)
    if not text or text.lower() in {"none", "null", "n/a", "unknown"}:
        return None
    return text


def _normalize_token(v: Any) -> str:
    text = _clean_text(v).lower()
    text = text.replace("-", "_").replace(" ", "_")
    return re.sub(r"[^a-z0-9_]+", "", text)


def _normalize_rarity(v: Any) -> str:
    text = _clean_text(v).lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z ]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    aliases = {
        "common": "common",
        "uncommon": "uncommon",
        "rare": "rare",
        "very rare": "very_rare",
        "veryrare": "very_rare",
        "legendary": "legendary",
        "artifact": "artifact",
        "artifacts": "artifact",
        "varies": "varies",
        "various": "varies",
        "unknown": "unknown",
    }
    if text in aliases:
        return aliases[text]
    raise ValueError(f"unknown rarity: {v!r}")


def _normalize_item_kind(v: Any) -> str:
    text = _clean_text(v).lower()
    text = text.replace("wonderous", "wondrous")
    text = text.replace("-", " ").replace("_", " ")
    if "wondrous" in text:
        return "wondrous_item"
    if "weapon" in text:
        return "weapon"
    if "armor" in text or "armour" in text:
        return "armor"
    if "ring" in text:
        return "ring"
    if "potion" in text or "elixir" in text:
        return "potion"
    if "scroll" in text:
        return "scroll"
    if "wand" in text:
        return "wand"
    if "rod" in text:
        return "rod"
    if "staff" in text:
        return "staff"
    return _normalize_token(text) or "other"


CleanText = Annotated[str, BeforeValidator(_clean_text), Hint("trimmed, single-spaced text")]
OptionalCleanText = Annotated[
    str | None,
    BeforeValidator(_clean_optional_text),
    Hint("trimmed text, or null when absent"),
]


class Rarity(StrEnum):
    COMMON = "common"
    UNCOMMON = "uncommon"
    RARE = "rare"
    VERY_RARE = "very_rare"
    LEGENDARY = "legendary"
    ARTIFACT = "artifact"
    VARIES = "varies"
    UNKNOWN = "unknown"


class ItemKind(StrEnum):
    WONDROUS_ITEM = "wondrous_item"
    WEAPON = "weapon"
    ARMOR = "armor"
    RING = "ring"
    POTION = "potion"
    SCROLL = "scroll"
    WAND = "wand"
    ROD = "rod"
    STAFF = "staff"
    OTHER = "other"


class EquipmentSlot(StrEnum):
    HEAD = "head"
    NECK = "neck"
    SHOULDERS = "shoulders"
    BODY = "body"
    HANDS = "hands"
    FEET = "feet"
    RING = "ring"
    SHIELD = "shield"
    WEAPON = "weapon"
    HELD = "held"
    CONSUMABLE = "consumable"
    CONTAINER = "container"
    INSTRUMENT = "instrument"
    NONE = "none"
    UNKNOWN = "unknown"


RarityValue = Annotated[Rarity, BeforeValidator(_normalize_rarity)]
ItemKindValue = Annotated[ItemKind, BeforeValidator(_normalize_item_kind)]
SlotValue = Annotated[EquipmentSlot, BeforeValidator(_normalize_token)]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Bonus(_Base):
    """A numeric or described bonus granted by an item."""

    target: CleanText = Field(description="What receives the bonus, e.g. AC, attack rolls.")
    value: OptionalCleanText = Field(default=None, description="Bonus amount, e.g. +1 or +2.")
    context: OptionalCleanText = Field(default=None, description="When or against whom it applies.")


class Defense(_Base):
    """Resistance, immunity, vulnerability, or defensive advantage."""

    kind: CleanText = Field(description="resistance, immunity, vulnerability, advantage, etc.")
    target: CleanText = Field(description="Damage type, condition, spell family, creature source, etc.")
    context: OptionalCleanText = None


class UsageLimit(_Base):
    """A limited-use rule such as charges, per-rest uses, or duration."""

    kind: CleanText = Field(description="charge, rest_reset, duration, cooldown, once_per_day, etc.")
    amount: OptionalCleanText = None
    reset: OptionalCleanText = None
    context: OptionalCleanText = None


class ItemMechanics(_Base):
    """Derived tags that make catalog search and pattern analysis easier."""

    equipment_slots: list[SlotValue] = Field(default_factory=list)
    bonuses: list[Bonus] = Field(default_factory=list)
    damage_types: list[CleanText] = Field(default_factory=list)
    defenses: list[Defense] = Field(default_factory=list)
    spells_granted: list[CleanText] = Field(default_factory=list)
    conditions_inflicted: list[CleanText] = Field(default_factory=list)
    target_creatures: list[CleanText] = Field(default_factory=list)
    environment_tags: list[CleanText] = Field(default_factory=list)
    usage_limits: list[UsageLimit] = Field(default_factory=list)
    action_economy: list[CleanText] = Field(default_factory=list)
    notes: list[CleanText] = Field(default_factory=list)


class SourceMagicItem(_Base):
    """Structured facts extracted from a PDF page.

    The source PDF text can be inspected through ExtractionPage.ocr_text using
    source_pages. Description is intentionally a concise, non-verbatim mechanics
    summary so the VLM extractor does not need to reproduce long page prose.
    """

    name: CleanText = Field(min_length=1)
    printed_type_line: CleanText = Field(
        description="The catalog's italic type/rarity line, copied from the page."
    )
    item_kind: ItemKindValue
    subtype: OptionalCleanText = Field(
        default=None,
        description="Printed subtype from parentheses, e.g. shield, plate, any axe.",
    )
    rarity: RarityValue = Rarity.UNKNOWN
    requires_attunement: bool = False
    attunement_requirement: OptionalCleanText = Field(
        default=None,
        description="Requirement after 'requires attunement by ...', if any.",
    )
    is_cursed: bool = False
    description: CleanText = Field(
        min_length=1,
        description=(
            "Concise non-verbatim summary of visible mechanics. Full OCR text "
            "is stored on ExtractionPage and linked by source_pages."
        ),
    )
    source_pages: list[int] = Field(default_factory=list)
    continues_from_previous_page: bool = False
    continues_on_next_page: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    warnings: list[CleanText] = Field(default_factory=list)

    @field_validator("source_pages")
    @classmethod
    def _source_pages_positive(cls, pages: list[int]) -> list[int]:
        return sorted({page for page in pages if page > 0})


class CatalogPageExtraction(_Base):
    """Raw VLM extraction result for one rendered page."""

    page_number: int = Field(gt=0)
    items: list[SourceMagicItem] = Field(default_factory=list)
    page_warnings: list[CleanText] = Field(default_factory=list)


class MagicItem(SourceMagicItem):
    """Final enriched ontology entity stored in Postgres."""

    id: UUID = Field(default_factory=uuid4)
    mechanics: ItemMechanics = Field(default_factory=ItemMechanics)


class ExtractionPage(_Base):
    """Auditable page-level artifact stored alongside item entities.

    This preserves the generated page image, full OCR text, and raw VLM payload
    from the PDF-driven extraction run.
    """

    page_number: int = Field(gt=0)
    image_path: CleanText
    ocr_text: CleanText = ""
    raw_vlm_json: dict[str, Any] = Field(default_factory=dict)
    status: CleanText = "pending"
    warnings: list[CleanText] = Field(default_factory=list)


REGISTRY: dict[str, type[BaseModel]] = {
    "SourceMagicItem": SourceMagicItem,
    "CatalogPageExtraction": CatalogPageExtraction,
    "ItemMechanics": ItemMechanics,
    "MagicItem": MagicItem,
    "ExtractionPage": ExtractionPage,
}
