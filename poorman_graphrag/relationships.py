"""Defines relationship types and metadata for connecting entities in a knowledge graph.

This module contains type definitions and models for representing different kinds of
relationships between entities in an academic knowledge graph. This includes both
document-level relationships (like citations and authorship) as well as content-level
relationships (like semantic similarity or causal relationships).

The relationships are designed to capture both explicit connections stated in the
source documents as well as implicit relationships that may be inferred through
analysis.
"""

import hashlib

import llamabot as lmb
from pydantic import BaseModel, Field

from poorman_graphrag.entities import Entity


class Relationship(BaseModel):
    """Represents a relationship between two entities.

    :param source: Source entity
    :param target: Target entity
    :param relation_type: Type of relationship
    :param metadata: Optional dictionary of additional metadata
    :param created_at: Timestamp of when relationship was created
    :param updated_at: Timestamp of when relationship was last updated
    """

    source: Entity = Field(..., description="Source entity")
    target: Entity = Field(..., description="Target entity")
    relation_type: str = Field(..., description="Type of relationship")
    summary: list[str] = Field(
        ..., description="Summary of relationship as described in the text."
    )
    quote: str = Field(
        ..., description="Quote from the text that supports the relationship"
    )

    def hash(self) -> str:
        """Generate hash for relationship."""
        return hashlib.sha256(
            f"{self.relation_type}:{self.source.hash()}:{self.target.hash()}".encode()
        ).hexdigest()

    def __add__(self, other: "Relationship") -> "Relationship":
        """Add two relationships together by combining their summaries.

        :param other: Another Relationship instance to combine with
        :return: New Relationship with combined summaries
        """
        if (
            self.source != other.source
            or self.target != other.target
            or self.relation_type != other.relation_type
        ):
            raise ValueError(
                "Can only add relationships with matching source, target and type"
            )

        combined_summary = list(set(self.summary + other.summary))
        return Relationship(
            source=self.source,
            target=self.target,
            relation_type=self.relation_type,
            summary=combined_summary,
        )


class Relationships(BaseModel):
    """Collection of relationships between entities.

    :param relationships: List of Relationship objects
    """

    relationships: list[Relationship] = Field(
        default_factory=list, description="List of relationships"
    )

    def update(self, **kwargs) -> "Relationships":
        """Return a new Relationships instance with updated relationships.

        :param kwargs: Fields to update on each relationship
        :return: New Relationships instance with updated relationships
        """
        updated = [r.update(**kwargs) for r in self.relationships]
        return Relationships(relationships=updated)

    def __iter__(self):
        """Iterate over relationships in the collection.

        :return: Iterator over Relationship objects
        """
        return iter(self.relationships)

    def __getitem__(self, idx: int) -> Relationship:
        """Get relationship at specified index.

        :param idx: Index of relationship to retrieve
        :return: Relationship at specified index
        """
        return self.relationships[idx]

    def __len__(self) -> int:
        """Get number of relationships in collection.

        :return: Number of relationships
        """
        return len(self.relationships)

    def __bool__(self) -> bool:
        """Check if collection contains any relationships.

        :return: True if collection contains relationships, False otherwise
        """
        return bool(self.relationships)

    def __add__(self, other: "Relationships") -> "Relationships":
        """Combine two relationship collections.

        :param other: Another Relationships instance to combine with
        :return: New Relationships instance with combined relationships
        """
        return Relationships(relationships=self.relationships + other.relationships)

    def __contains__(self, item: Relationship) -> bool:
        """Check if relationship exists in collection.

        :param item: Relationship to check for
        :return: True if relationship exists in collection, False otherwise
        """
        return item in self.relationships


@lmb.prompt("system")
def relationship_extractor_prompt():
    """You are an expert at extracting relationships from text. Given a chunk of text,
    identify relationships between entities mentioned in the text."""


def get_relationship_extractor() -> lmb.StructuredBot:
    """Get a relationship extractor."""
    return lmb.StructuredBot(
        pydantic_model=Relationships,
        system_prompt=relationship_extractor_prompt(),
        model_name="gpt-4o",
    )


@lmb.prompt("user")
def relationship_extractor_user_prompt(chunk_text: str, existing_entities: str) -> str:
    """Here is a chunk of text to process:

    {{ chunk_text }}

    -----

    And here are existing entities:

    {{ existing_entities }}
    """
