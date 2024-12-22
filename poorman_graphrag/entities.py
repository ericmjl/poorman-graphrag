"""Entity types for biomedical, statistical, and machine learning literature.

This module defines entity types that can be used to structure knowledge graphs
from scientific literature in biomedical sciences, statistics, and machine learning.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator
from thefuzz import fuzz

# TODO: Make EntityType configurable by a user.
# In chatting with Claude, I have a few strategies.
# These are documented in docs/configuration/entity_types.md

# Research entities
ResearchEntityType = Literal["paper", "author", "institution", "dataset"]

# Biomedical entities
BiomedicalEntityType = Literal[
    "gene", "protein", "disease", "drug", "pathway", "cell_type", "organism"
]

# Statistical/ML concepts
StatMLEntityType = Literal[
    "model", "algorithm", "metric", "hypothesis", "statistical_test", "distribution"
]

# Combined entity type
EntityType = Literal[
    ResearchEntityType,
    BiomedicalEntityType,
    StatMLEntityType,
]


class Entity(BaseModel):
    """Base class for entities in the knowledge graph.

    :param entity_type: The type of entity
    :param name: Name or title of the entity
    :param description: Optional description of the entity
    :param metadata: Optional dictionary of additional metadata
    """

    entity_type: EntityType = Field(..., description="Type of entity")
    name: str = Field(..., description="Name or title of entity")
    description: Optional[str] = Field(None, description="Description of entity")
    quote: str = Field(
        ...,
        description=(
            "Quote from the text that contains the entity. "
            "It should be longer than the name and include surrounding context."
        ),
    )

    @model_validator(mode="after")
    def validate_quote_length(self) -> "Entity":
        """Validate that the quote is longer than the name."""
        if len(self.quote) <= len(self.name):
            raise ValueError("Quote must be longer than the entity name")
        return self

    def update(self, **kwargs) -> "Entity":
        """Return a new Entity instance with updated fields.

        :param kwargs: Fields to update
        :return: New Entity instance
        """
        return Entity(**{**self.model_dump(), **kwargs, "updated_at": datetime.now()})


class Entities(BaseModel):
    """Collection of Entity objects for batch processing.

    :param entities: List of Entity objects
    """

    entities: List[Entity] = Field(
        default_factory=list, description="List of Entity objects"
    )

    def __iter__(self):
        """Allow iteration over entities."""
        return iter(self.entities)

    def __len__(self):
        """Return number of entities."""
        return len(self.entities)

    def __getitem__(self, idx):
        """Allow indexing and slicing to get entities.

        :param idx: Integer index or slice object
        :return: Single Entity if integer index, new Entities instance if slice
        """
        if isinstance(idx, slice):
            return Entities(entities=self.entities[idx])
        return self.entities[idx]

    def filter_by_type(self, entity_type: EntityType) -> "Entities":
        """Return new Entities instance with only entities of specified type.

        :param entity_type: Type to filter by
        :return: New Entities instance with filtered entities
        """
        filtered = [e for e in self.entities if e.entity_type == entity_type]
        return Entities(entities=filtered)

    def update(self, **kwargs) -> "Entities":
        """Return new Entities instance with all entities updated.

        :param kwargs: Fields to update on each entity
        :return: New Entities instance with updated entities
        """
        updated = [e.update(**kwargs) for e in self.entities]
        return Entities(entities=updated)

    @classmethod
    def from_jsonl(cls, path: Path) -> "Entities":
        """Create Entities instance from a JSONL file.

        :param path: Path to JSONL file containing entity data
        :return: New Entities instance
        """
        entities = []
        with path.open() as f:
            for line in f:
                entity_data = json.loads(line)
                entities.append(Entity(**entity_data))
        return cls(entities=entities)

    def to_jsonl(self, path: Path) -> None:
        """Save entities to a JSONL file.

        :param path: Path to save JSONL file
        """
        with path.open("w") as f:
            for entity in self.entities:
                f.write(json.dumps(entity.model_dump()) + "\n")

    def to_dict(self) -> Dict[str, Any]:
        """Convert entities to dictionary format.

        :return: Dictionary representation of entities
        """
        return {"entities": [e.model_dump() for e in self.entities]}

    def deduplicate(
        self, merge_funcs: Optional[List[Callable[["Entities"], "Entities"]]] = None
    ) -> "Entities":
        """Merge similar entities by applying a sequence of deduplication functions.

        By default, applies merge_exact_duplicates()
        followed by merge_levenshtein_similar().

        Custom merge functions can be provided
        that take an Entities object and return a new Entities object.

        For merge functions that require additional arguments,
        use functools.partial to create
        a function that only takes Entities as input. For example:

        ```python
        from functools import partial
        custom_levenshtein = partial(merge_levenshtein_similar, similarity_threshold=80)
        entities.deduplicate(merge_funcs=[merge_exact_duplicates, custom_levenshtein])
        ```

        :param merge_funcs: List of functions that each take Entities
            and return Entities.
            If None, uses default merge functions.
        :return: New Entities instance with merged entities
        """
        if merge_funcs is None:
            merge_funcs = [merge_exact_duplicates, merge_levenshtein_similar]

        result = self
        for merge_func in merge_funcs:
            result = merge_func(result)
        return result


def merge_exact_duplicates(entities: "Entities") -> "Entities":
    """Merge entities that have exactly the same name (ignoring case) and type.

    :param entities: Entities instance to deduplicate
    :return: New Entities instance with exact duplicates merged
    """
    # Group entities by type and normalized name
    groups: Dict[tuple, list] = defaultdict(list)
    for entity in entities.entities:
        key = (entity.entity_type.lower().strip(), entity.name.lower().strip())
        groups[key].append(entity)

    # Merge entities in each group
    merged = []
    for group in groups.values():
        if len(group) == 1:
            merged.append(group[0])
        else:
            # Combine descriptions
            descriptions = [e.description for e in group if e.description]
            merged_desc = " ".join(descriptions) if descriptions else None
            # Use the first entity as base and update description
            merged.append(group[0].update(description=merged_desc))

    return Entities(entities=merged)


def merge_levenshtein_similar(
    entities: "Entities", similarity_threshold: int = 95
) -> "Entities":
    """Merge entities that have similar names according to fuzzy string matching.

    Criteria for merging:
    - Entities must have the same type
    - Entities must have a similarity score greater
      than or equal to similarity_threshold

    :param entities: Entities instance to deduplicate
    :param similarity_threshold: Minimum similarity ratio (0-100)
        to consider entities as similar
    :return: New Entities instance with similar entities merged
    """

    # Group entities by type first
    type_groups: Dict[str, list] = defaultdict(list)
    for entity in entities.entities:
        type_groups[entity.entity_type.lower().strip()].append(entity)

    # For each type group, find and merge similar entities
    final_entities = []
    for type_group in type_groups.values():
        # Keep track of which entities have been merged
        merged = set()

        for i, entity1 in enumerate(type_group):
            if i in merged:
                continue

            similar_group = [entity1]

            # Compare with remaining entities
            for j, entity2 in enumerate(type_group[i + 1 :], start=i + 1):
                if j in merged:
                    continue

                # Use token sort ratio to handle word reordering
                similarity = fuzz.token_sort_ratio(
                    entity1.name.lower(), entity2.name.lower()
                )
                if similarity >= similarity_threshold:
                    similar_group.append(entity2)
                    merged.add(j)

            if len(similar_group) == 1:
                final_entities.append(entity1)
            else:
                # Merge descriptions
                descriptions = [e.description for e in similar_group if e.description]
                merged_desc = ", ".join(descriptions) if descriptions else None
                # Use the first entity as base and update description
                final_entities.append(similar_group[0].update(description=merged_desc))
                merged.add(i)

    return Entities(entities=final_entities)
