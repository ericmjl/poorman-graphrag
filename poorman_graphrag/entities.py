"""Entity types for biomedical, statistical, and machine learning literature.

This module defines entity types that can be used to structure knowledge graphs
from scientific literature in biomedical sciences, statistics, and machine learning.
"""

import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal

import llamabot as lmb
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
    :param summary: Summary paragraph about the entity
    """

    entity_type: EntityType = Field(..., description="Type of entity")
    name: str = Field(..., description="Name or title of entity")
    summary: List[str] = Field(..., description="Summary sentence(s) about entity")

    @model_validator(mode="after")
    def validate_summary(self) -> "Entity":
        """Validate that the summary is not empty."""
        if not self.summary:
            raise ValueError("Summary cannot be empty")
        return self

    def __hash__(self) -> int:
        """Generate hash for entity based on type and name."""
        hasher = hashlib.sha256()
        hasher.update(f"{self.entity_type.lower()}:{self.name.lower()}".encode())
        return int.from_bytes(hasher.digest(), "big")

    def hash(self) -> str:
        """Generate hash for entity based on type and name."""
        # use sha256
        hasher = hashlib.sha256()
        hasher.update(f"{self.entity_type.lower()}:{self.name.lower()}".encode())
        return hasher.hexdigest()

    def update(self, **kwargs) -> "Entity":
        """Return a new Entity instance with updated fields.

        :param kwargs: Fields to update
        :return: New Entity instance
        """
        return Entity(**{**self.model_dump(), **kwargs, "updated_at": datetime.now()})

    def __add__(self, other: "Entity") -> "Entity":
        """Merge this entity with another entity.

        :param other: Another entity to merge with
        :return: New Entity instance with merged data
        """
        # Combine summaries if they exist
        summaries = set()
        if self.summary:
            summaries.update(self.summary)
        if other.summary:
            summaries.update(other.summary)

        return Entity(
            entity_type=self.entity_type,
            name=self.name,  # Keep the name of the primary entity
            summary=list(summaries),
        )

    def __radd__(self, other: "Entity") -> "Entity":
        """Right-hand addition, called when doing other + self.

        :param other: Another entity to merge with
        :return: New Entity instance with merged data
        """
        return self + other  # Reuse __add__ implementation

    def __lt__(self, other: "Entity") -> bool:
        """Compare entities based on their hash strings.

        :param other: Another entity to compare with
        :return: True if this entity's hash is less than other's hash
        """
        return self.hash() < other.hash()

    def __gt__(self, other: "Entity") -> bool:
        """Compare entities based on their hash strings.

        :param other: Another entity to compare with
        :return: True if this entity's hash is greater than other's hash
        """
        return self.hash() > other.hash()


class Entities(BaseModel):
    """Collection of Entity objects for batch processing.

    :param entities: List of Entity objects or their dictionary representations
    """

    entities: List[Entity] = Field(
        default_factory=list, description="List of Entity objects"
    )

    @model_validator(mode="before")
    def validate_entities(cls, values):
        """Convert dictionaries to Entity objects if needed."""
        if "entities" in values:
            raw_entities = values["entities"]
            converted = []
            for entity in raw_entities:
                if isinstance(entity, dict):
                    converted.append(Entity(**entity))
                elif isinstance(entity, Entity):
                    converted.append(entity.model_dump())
                else:
                    raise ValueError(f"Invalid entity type: {type(entity)}")
            values["entities"] = converted
        return values

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

    def names(self, with_types: bool = False) -> List[str]:
        """Return list of entity names.

        :param with_types: Whether to include entity types in the output
        :return: List of entity names
        """
        if with_types:
            return [f"{e.entity_type}: {e.name}" for e in self.entities]
        return [e.name for e in self.entities]

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


def identify_levenshtein_similar(
    entities: "Entities", similarity_threshold: int = 95
) -> Dict[tuple, List[Entity]]:
    """Identify entities that have similar names according to fuzzy string matching.

    :param entities: Entities instance to check for similar names
    :param similarity_threshold: Minimum similarity ratio (0-100) for names
        to be considered similar
    :return: Dictionary mapping (type, name) tuples to lists of similar entities
    """
    # Group entities by type first to avoid comparing entities of different types
    type_groups: Dict[str, List[Entity]] = defaultdict(list)
    for entity in entities.entities:
        type_groups[entity.entity_type.lower().strip()].append(entity)

    similar_groups: Dict[tuple, List[Entity]] = defaultdict(list)

    # For each type group, compare all pairs of entities
    for entity_type, type_group in type_groups.items():
        # Keep track of which entities have been grouped
        processed = set()

        for i, entity1 in enumerate(type_group):
            if i in processed:
                continue

            # Start a new group with this entity
            key = (entity_type, entity1.name.lower().strip())
            group = [entity1]

            # Compare with all other unprocessed entities of same type
            for j, entity2 in enumerate(type_group[i + 1 :], start=i + 1):
                if j in processed:
                    continue

                similarity = fuzz.token_sort_ratio(
                    entity1.name.lower(), entity2.name.lower()
                )

                if similarity >= similarity_threshold:
                    group.append(entity2)
                    processed.add(j)

            # Only add groups with more than one entity
            if len(group) > 1:
                similar_groups[key] = group
                processed.add(i)

    return similar_groups


@lmb.prompt("system")
def entity_extractor_prompt():
    """You are an expert at extracting entities from text. Given a chunk of text,
    identify entities mentioned in the text. You will be optionally provided with
    a list of existing entities that can be reused."""


@lmb.prompt("user")
def entity_extractor_user_prompt(text: str, existing_entities: str):
    """Here is a chunk of text to process:

    {{ text }}

    -----

    And here are existing entities:

    {{ existing_entities }}
    """


def get_entity_extractor(**kwargs) -> lmb.StructuredBot:
    """Get an entity extractor.

    :return: Entity extractor
    """
    system_prompt = kwargs.pop("system_prompt", entity_extractor_prompt())
    pydantic_model = kwargs.pop("pydantic_model", Entities)
    model_name = kwargs.pop("model_name", "gpt-4o")
    return lmb.StructuredBot(
        system_prompt=system_prompt,
        pydantic_model=pydantic_model,
        model_name=model_name,
        **kwargs,
    )


###
###

# def identify_exact_duplicates(entities: "Entities") -> Dict[tuple, list]:
#     """Identify entities that have exactly the same name (ignoring case) and type.

#     :param entities: Entities instance to check for duplicates
#     :return: Dictionary mapping (type, name) tuples to lists of duplicate entities
#     """
#     groups: Dict[tuple, list] = defaultdict(list)
#     for entity in entities.entities:
#         key = (entity.entity_type.lower().strip(), entity.name.lower().strip())
#         groups[key].append(entity)

#     # Filter out groups with no duplicates
#     return {k: v for k, v in groups.items() if len(v) > 1}


# def merge_entity_group(group: List[Entity]) -> tuple[Entity, Dict[str, str]]:
#     """Merge a group of duplicate entities into a single entity.

#     :param group: List of entities to merge
#     :return: Tuple of (merged entity, mapping of old->new entity names)
#     """
#     merge_mapping: Dict[str, str] = {}
#     result = group[0]

#     for other in group[1:]:
#         merge_mapping[other.name] = result.name
#         result = other + result  # Uses __add__

#     return result, merge_mapping


# def merge_levenshtein_similar(
#     entities: "Entities", similarity_threshold: int = 95
# ) -> tuple["Entities", Dict[str, str]]:
#     """Merge entities that have similar names according to fuzzy string matching.

#     :param entities: Entities instance to deduplicate
#     :param similarity_threshold: Minimum similarity ratio (0-100)
#     :return: Tuple of (new Entities instance, mapping of old->new entity names)
#     """
#     # Group entities by type first
#     type_groups: Dict[str, list] = defaultdict(list)
#     for entity in entities.entities:
#         type_groups[entity.entity_type.lower().strip()].append(entity)

#     merge_mapping: Dict[str, str] = {}
#     final_entities = []

#     for type_group in type_groups.values():
#         merged = set()

#         for i, entity1 in enumerate(type_group):
#             if i in merged:
#                 continue

#             similar_group = [entity1]

#             for j, entity2 in enumerate(type_group[i + 1 :], start=i + 1):
#                 if j in merged:
#                     continue

#                 similarity = fuzz.token_sort_ratio(
#                     entity1.name.lower(), entity2.name.lower()
#                 )
#                 if similarity >= similarity_threshold:
#                     similar_group.append(entity2)
#                     merged.add(j)

#             if len(similar_group) == 1:
#                 final_entities.append(entity1)
#             else:
#                 # Merge all entities in the group
#                 result = similar_group[0]
#                 for other in similar_group[1:]:
#                     merge_mapping[other.name] = result.name
#                     result = result.merge_with(other)
#                 final_entities.append(result)
#                 merged.add(i)

#     return Entities(entities=final_entities), merge_mapping
