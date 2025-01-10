"""Communities data structure for GraphRAG."""

from hashlib import sha256
from typing import List

import llamabot as lmb
from pydantic import BaseModel, Field

from poorman_graphrag.entities import Entity


class Community(BaseModel):
    """A community of entities in the graph.

    :ivar entities: List of Entity objects that belong to this community
    :ivar summary: Summary description of the community
    """

    entities: List[Entity] = Field(
        description="List of Entity objects in the community"
    )
    summary: str = Field(description="Summary of the community")

    def hash(self) -> str:
        """Generate a unique hash for this community based on its entities.

        :return: SHA256 hash of the sorted and joined entity hashes
        """
        # Sort entities by their hashes to ensure consistent hashing
        sorted_entities = sorted([e.hash() for e in self.entities])
        community_string = "-".join(sorted_entities)
        return sha256(community_string.encode()).hexdigest()


class Communities(BaseModel):
    """Collection of communities in the graph.

    :ivar communities: List of Community objects
    """

    communities: List[Community] = Field(
        default_factory=list, description="List of communities in the graph"
    )


class CommunitySummary(BaseModel):
    """Summary of a community of nodes.

    :ivar summary: A summary of the community of nodes
    """

    summary: str = Field(description="A summary of the community of nodes.")


@lmb.prompt("system")
def community_summarizer_system_prompt():
    """You are a helpful assistant that summarizes communities of nodes in a graph."""


def get_community_summarizer(**kwargs) -> lmb.StructuredBot:
    """Get a community summarizer bot.

    :param model_name: Name of the model to use for summarization
    :param system_prompt: System prompt for the summarizer
    :return: A StructuredBot instance for summarizing communities
    """
    model_name = kwargs.pop("model_name", "gpt-4o")
    system_prompt = kwargs.pop("system_prompt", community_summarizer_system_prompt())
    pydantic_model = kwargs.pop("pydantic_model", CommunitySummary)
    community_summarizer = lmb.StructuredBot(
        model_name=model_name,
        pydantic_model=pydantic_model,
        system_prompt=system_prompt,
        **kwargs,
    )
    return community_summarizer


@lmb.prompt("user")
def community_content(nodes: list, edges: list) -> str:
    """
    Here are the relations in the community:

    {% for edge in edges %}
    {{ edge[0] }} --{{ edge[2].get('relation_type', '') }}--> {{ edge[1] }}
    {% endfor %}

    Here are the nodes in the community:

    {% for node_id, node_data in nodes %}
    ({{ node_data.get('entity_type', '') }}) {{ node_data.get('name', '') }}: {{ node_data.get('summary', '') }}
    {% endfor %}
    """  # noqa: E501
