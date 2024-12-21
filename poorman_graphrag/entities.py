"""Entity types for biomedical, statistical, and machine learning literature.

This module defines entity types that can be used to structure knowledge graphs
from scientific literature in biomedical sciences, statistics, and machine learning.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Dict, Any

from pydantic import BaseModel, Field, model_validator

# Research entities
ResearchEntityType = Literal["paper", "author", "institution", "dataset"]

# Biomedical entities
BiomedicalEntityType = Literal["gene", "protein", "disease", "drug", "pathway", "cell_type", "organism"]

# Statistical/ML concepts
StatMLEntityType = Literal["model", "algorithm", "metric", "hypothesis", "statistical_test", "distribution"]

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
    quote: str = Field(..., description="Quote from the text that contains the entity. It should be longer than the name and include surrounding context.")

    @model_validator(mode='after')
    def validate_quote_length(self) -> 'Entity':
        """Validate that the quote is longer than the name."""
        if len(self.quote) <= len(self.name):
            raise ValueError("Quote must be longer than the entity name")
        return self

    def update(self, **kwargs) -> "Entity":
        """Return a new Entity instance with updated fields.

        :param kwargs: Fields to update
        :return: New Entity instance
        """
        return Entity(
            **{
                **self.dict(),
                **kwargs,
                "updated_at": datetime.now()
            }
        )



class Entities(BaseModel):
    """Collection of Entity objects for batch processing.

    :param entities: List of Entity objects
    """
    entities: List[Entity] = Field(default_factory=list, description="List of Entity objects")

    def __iter__(self):
        """Allow iteration over entities."""
        return iter(self.entities)

    def __len__(self):
        """Return number of entities."""
        return len(self.entities)

    def __getitem__(self, idx):
        """Allow indexing to get entities."""
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
        with path.open('w') as f:
            for entity in self.entities:
                f.write(json.dumps(entity.model_dump()) + '\n')

    def to_dict(self) -> Dict[str, Any]:
        """Convert entities to dictionary format.

        :return: Dictionary representation of entities
        """
        return {"entities": [e.model_dump() for e in self.entities]}


# Add paper metadata relationship types
PaperRelationType = Literal[
    # Document relationships
    "author_of",         # Person is author of document
    "cites",            # Document cites another document
    "published_in",     # Document published in journal/venue
    "affiliated_with",  # Author affiliated with institution
    "funded_by",       # Research funded by organization/grant
    "keywords",        # Document has associated keywords
    "published_on",    # Publication date
    "reviews",         # Document reviews another document
    "translates",      # Document translates another document
    "retracts",        # Document retracts another document
    "supplements",     # Document supplements another document
    "contributes_to",  # Author contributes to document
    "edits",          # Editor edits document
    "peer_reviews",    # Reviewer peer reviews document
    "supervises",      # Supervisor oversees research
    "corresponds_for", # Corresponding author for document
]
