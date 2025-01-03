"""Entity deduplication utilities for GraphRAG.

This module provides functionality for determining
whether two entities in a knowledge graph
should be considered the same entity and merged together.
It includes:

- A Pydantic model for structured entity comparison results
- A prompt-based function for comparing entities
- A factory function for creating entity comparison bots
"""

import llamabot as lmb
from pydantic import BaseModel, Field


class IsSameEntity(BaseModel):
    """Structured output for entity comparison results.

    This model represents the result of comparing two entities to determine if they
    refer to the same underlying entity in the knowledge graph. It includes both
    the boolean determination and the reasoning behind it.

    :ivar is_same_entity: Whether the entities are semantically the same entity
    :ivar reason: The reason for the determination
    """

    is_same_entity: bool = Field(
        description="Whether the entities are semantically the same entity"
    )
    reason: str = Field(description="The reason for the answer")

    def __bool__(self):
        """Convert to boolean.

        :return: The is_same_entity value
        """
        return self.is_same_entity


@lmb.prompt("user")
def is_same_entity(entities) -> IsSameEntity:
    """Here are the entities:

    {% for entity in entities %}- {{ entity.entity_type }}: {{ entity.name }}
    {% endfor %}
    """


def get_entity_judge(
    system_prompt: str = "You are a judge of whether two entities in a knowledge graph "
    "are similar enough to be considered the same entity. ",
    model_name: str = "gpt-4o",
) -> lmb.StructuredBot:
    """Get a StructuredBot that judges whether two entities are the same.

    :param system_prompt: The system prompt to use for the bot
    :param model_name: The model name to use for the bot
    :return: A StructuredBot configured to judge entity similarity
    """
    return lmb.StructuredBot(
        system_prompt=lmb.system(system_prompt),
        pydantic_model=IsSameEntity,
        model_name=model_name,
    )
