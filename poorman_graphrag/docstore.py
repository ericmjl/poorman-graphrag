"""Knowledge Graph document store for LlamaBot."""

from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional, Set

import llamabot as lmb
import networkx as nx
from chonkie import SDPMChunker
from llamabot.components.docstore import AbstractDocumentStore
from networkx.algorithms.community import louvain_communities

from poorman_graphrag.communities import (
    Communities,
    Community,
    community_content,
    get_community_summarizer,
)
from poorman_graphrag.entities import Entities, Entity, identify_levenshtein_similar
from poorman_graphrag.entity_deduplication import get_entity_judge, is_same_entity
from poorman_graphrag.relationships import Relationship, Relationships


class KnowledgeGraphDocStore(AbstractDocumentStore):
    """Knowledge Graph document store for LlamaBot.

    :param storage_path: Path to JSON file for persistent storage
    """

    def __init__(
        self, storage_path: Path | str = Path.home() / ".llamabot" / "kg_docstore.json"
    ):
        self.storage_path = Path(storage_path)
        self.doc_index: Dict[str, str] = {}
        self.chunk_index: Dict[str, str] = {}
        self.entity_index: Dict[str, Entity] = {}
        self.relation_index: Dict[str, Relationship] = {}
        self.community_index: Dict[str, Community] = {}
        self.doc_chunk_links: Dict[str, Set[str]] = {}
        self.chunk_entity_links: Dict[str, Set[str]] = {}
        self.chunk_relation_links: Dict[str, Set[str]] = {}
        self.entity_chunk_links: Dict[str, Set[str]] = {}
        self.entity_community_links: Dict[str, Set[str]] = {}

        # Load existing data if available
        if self.storage_path.exists():
            self._load()

        # Flag to prevent saving during extend operations
        self._in_extend = False

        # Instantiate chunker
        self.chunker = SDPMChunker()

    def add_document(self, text: str) -> str:
        """Add document to docstore.

        :param text: Document text
        :return: Document hash
        """
        doc_hash = sha256(text.encode()).hexdigest()
        self.doc_index[doc_hash] = text
        self.doc_chunk_links[doc_hash] = set()
        self._save()
        return doc_hash

    def add_chunk(self, doc_hash: str, chunk_text: str) -> str:
        """Add chunk to docstore and link to document.

        :param doc_hash: Hash of parent document
        :param chunk_text: Chunk text
        :return: Chunk hash
        """
        if doc_hash not in self.doc_index:
            raise ValueError(f"Document with hash {doc_hash} not found")

        chunk_hash = sha256(chunk_text.encode()).hexdigest()
        self.chunk_index[chunk_hash] = chunk_text
        self.chunk_entity_links[chunk_hash] = set()
        self.chunk_relation_links[chunk_hash] = set()
        self.doc_chunk_links[doc_hash].add(chunk_hash)
        self._save()
        return chunk_hash

    def add_entities(self, chunk_hash: str, entities: Entities) -> List[str]:
        """Add entities to docstore and link to chunk.
        If an entity with the same hash already exists,
        merges the new entity with the existing one.

        :param chunk_hash: Hash of parent chunk
        :param entities: List of Entity objects
        :return: List of entity hashes
        """
        if chunk_hash not in self.chunk_index:
            raise ValueError(f"Chunk with hash {chunk_hash} not found")

        entity_hashes = []
        for entity in entities:
            entity_hash = entity.hash()
            if entity_hash in self.entity_index:
                # Merge with existing entity
                self.entity_index[entity_hash] = self.entity_index[entity_hash] + entity
            else:
                self.entity_index[entity_hash] = entity

            # Set up entity-chunk links
            if entity_hash not in self.entity_chunk_links:
                self.entity_chunk_links[entity_hash] = set()
            self.entity_chunk_links[entity_hash].add(chunk_hash)

            self.chunk_entity_links[chunk_hash].add(entity_hash)
            entity_hashes.append(entity_hash)

        self._save()
        return entity_hashes

    def add_relations(self, chunk_hash: str, relations: Relationships) -> List[str]:
        """Add relations to docstore and link to chunk.
        Also adds any new entities found in the relations.

        :param chunk_hash: Hash of parent chunk
        :param relations: List of Relation objects
        :return: List of relation hashes
        """
        if chunk_hash not in self.chunk_index:
            raise ValueError(f"Chunk with hash {chunk_hash} not found")

        # First collect all entities from relations
        new_entities = []
        for relation in relations:
            new_entities.extend([relation.source, relation.target])

        # Add any new entities
        self.add_entities(chunk_hash, new_entities)

        # Now add the relations
        relation_hashes = []
        for relation in relations:
            relation_hash = relation.hash()
            self.relation_index[relation_hash] = relation
            self.chunk_relation_links[chunk_hash].add(relation_hash)
            relation_hashes.append(relation_hash)

        self._save()
        return relation_hashes

    def add_communities(self, communities: Communities) -> List[str]:
        """Add communities to the docstore.

        :param communities: Communities object containing communities to add
        :return: List of community hashes
        """
        community_hashes = []
        for community in communities.communities:
            community_hash = community.hash()
            self.community_index[community_hash] = community
            community_hashes.append(community_hash)

            # Update entity_community_links for each entity in the community
            for entity in community.entities:
                if entity.hash() in self.entity_index:  # Only link if it's an entity
                    if entity.hash() not in self.entity_community_links:
                        self.entity_community_links[entity.hash()] = set()
                    self.entity_community_links[entity.hash()].add(community_hash)

        self._save()
        return community_hashes

    def append(self, document: str):
        """Append a document to the docstore."""
        # Firstly, add the document
        doc_hash = self.add_document(document)

        # Then, chunk the document and add the chunks
        chunks = self.chunker.chunk(document)
        for chunk in chunks:
            self.add_chunk(doc_hash, chunk.text)

        # Now, extract entities from the chunks
        entity_extractor = lmb.StructuredBot(
            system_prompt=(
                "You are an expert at extracting entities from text. Given a chunk of "
                "text, identify entities mentioned in the text. "
                "You will be optionally provided with a list of existing entities that "
                "can be reused."
            ),
            pydantic_model=Entities,
        )
        for chunk in chunks:
            chunk_hash = sha256(chunk.text.encode()).hexdigest()
            existing_entities = "\n".join(
                [e.model_dump_json() for e in self.entity_index.values()]
            )
            entities = entity_extractor(
                lmb.user(
                    "Here is chunk to process:\n",
                    chunk.text,
                    "\n-----\n",
                    "\nHere are existing entities:\n",
                    existing_entities,
                )
            )
            self.add_entities(chunk_hash, entities)

        # Now, extract entities and relations from the chunks
        relationship_extractor = lmb.StructuredBot(
            system_prompt=(
                "You are an expert at extracting relationships between "
                "entities in text. Given a chunk of text, identify relationships "
                "between entities mentioned in the text. You will be optionally "
                "provided with a list of existing entities that can be reused. "
                "Reuse entities where possible. "
            ),
            pydantic_model=Relationships,
        )

        for chunk in chunks:
            existing_entities = "\n".join(
                [e.model_dump_json() for e in self.entity_index.values()]
            )
            chunk_hash = sha256(chunk.text.encode()).hexdigest()
            relationships = relationship_extractor(
                lmb.user(
                    "Text chunk:\n",
                    chunk.text,
                    "\n-----\n",
                    "Existing entities:\n",
                    existing_entities,
                )
            )
            self.add_relations(chunk_hash, relationships)

        # Now, deduplicate entities.
        similar_entities = identify_levenshtein_similar(self.entities)
        same_entity_judge = get_entity_judge()
        entity_groups_to_deduplicate = {}
        for entity_type, entities in similar_entities.items():
            result = same_entity_judge(is_same_entity(entities))
            if result:
                entity_groups_to_deduplicate[entity_type] = entities

        self.deduplicate_entities(entity_groups_to_deduplicate)

        # Identify communities in the graph and add them to the docstore.
        G = build_network(self)
        communities = louvain_communities(G.to_undirected())
        communities_to_add = []
        community_summarizer = get_community_summarizer()
        for community in communities:
            content = community_content(
                G.subgraph(community).nodes(data=True),
                G.subgraph(community).edges(data=True),
            )
            community_summary_response = community_summarizer(content)
            print("--------------------------------")
            print(G.subgraph(community).nodes(data=True))
            print("--------------------------------")
            # Convert node data back into Entity objects
            community_entities = []
            for node_id, node_data in G.subgraph(community).nodes(data=True):
                if node_data.get("type") == "entity":
                    entity = Entity(
                        entity_type=node_data["entity_type"],
                        name=node_data["name"],
                        summary=node_data["summary"],
                    )
                    community_entities.append(entity)

            communities_to_add.append(
                Community(
                    entities=Entities(
                        entities=[e.model_dump() for e in community_entities]
                    ),
                    summary=community_summary_response.summary,
                )
            )
        self.add_communities(
            Communities(communities=[c.model_dump() for c in communities_to_add])
        )

        self._save()

    # Just some notes for myself right now:
    # 1. I'm quite uncomfortable with how this DocStore contains StructuredBots.
    #    This feels like it should be separated out.
    # 2. Maybe the docstore should just accept a NetworkX graph instead.

    @property
    def entities(self) -> Entities:
        """Get all entities in the docstore."""
        return Entities(entities=list(self.entity_index.values()))

    def retrieve(
        self,
        chunks: Optional[Set[str]] = None,
        documents: Optional[Set[str]] = None,
        relations: Optional[Set[str]] = None,
        entities: Optional[Set[str]] = None,
        communities: Optional[Set[str]] = None,
        keywords: Optional[str] = None,
        n_results: int = 10,
    ) -> Dict[str, Set]:
        """Retrieve items from the docstore.

        :param chunks: Set of chunk hashes to retrieve
        :param documents: Set of document hashes to retrieve
        :param relations: Set of relation hashes to retrieve
        :param entities: Set of entity hashes to retrieve
        :param communities: Set of community hashes to retrieve
        :param keywords: Keywords to search across all text
        :param n_results: Number of results to return for keyword search
        :return: Dictionary mapping item types to sets of retrieved items
        """
        results = {
            "chunks": set(),
            "documents": set(),
            "relations": set(),
            "entities": set(),
            "communities": set(),
        }

        # Direct hash lookups
        if chunks:
            results["chunks"].update(
                self.chunk_index[h] for h in chunks if h in self.chunk_index
            )
        if documents:
            results["documents"].update(
                self.doc_index[h] for h in documents if h in self.doc_index
            )
        if relations:
            results["relations"].update(
                self.relation_index[h] for h in relations if h in self.relation_index
            )
        if entities:
            results["entities"].update(
                self.entity_index[h] for h in entities if h in self.entity_index
            )
        if communities:
            results["communities"].update(
                self.community_index[h]
                for h in communities
                if h in self.community_index
            )

        # Keyword search using BM25
        if keywords:
            from rank_bm25 import BM25Okapi

            # Search across all text content
            all_text = list(self.chunk_index.values()) + list(self.doc_index.values())
            tokenized_corpus = [doc.split() for doc in all_text]
            bm25 = BM25Okapi(tokenized_corpus)

            tokenized_query = keywords.split()
            doc_scores = bm25.get_scores(tokenized_query)

            # Get top n_results
            top_indices = sorted(
                range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True
            )[:n_results]

            for idx in top_indices:
                if idx < len(self.chunk_index):
                    results["chunks"].add(all_text[idx])
                else:
                    results["documents"].add(all_text[idx])

        return results

    def reset(self) -> None:
        """Reset the document store."""
        self.__init__(storage_path=self.storage_path)
        self._save()

    def _save(self) -> None:
        """Save the docstore to disk."""
        import json

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "doc_index": self.doc_index,
            "chunk_index": self.chunk_index,
            "entity_index": {k: v.model_dump() for k, v in self.entity_index.items()},
            "relation_index": {
                k: v.model_dump() for k, v in self.relation_index.items()
            },
            "community_index": {
                k: v.model_dump() for k, v in self.community_index.items()
            },
            "doc_chunk_links": {k: list(v) for k, v in self.doc_chunk_links.items()},
            "chunk_entity_links": {
                k: list(v) for k, v in self.chunk_entity_links.items()
            },
            "chunk_relation_links": {
                k: list(v) for k, v in self.chunk_relation_links.items()
            },
            "entity_chunk_links": {
                k: list(v) for k, v in self.entity_chunk_links.items()
            },
            "entity_community_links": {
                k: list(v) for k, v in self.entity_community_links.items()
            },
        }

        with self.storage_path.open("w") as f:
            json.dump(data, f)

    def _load(self) -> None:
        """Load the docstore from disk."""
        import json

        with self.storage_path.open("r") as f:
            data = json.load(f)

        self.doc_index = data["doc_index"]
        self.chunk_index = data["chunk_index"]
        self.entity_index = {k: Entity(**v) for k, v in data["entity_index"].items()}
        self.relation_index = {
            k: Relationship(**v) for k, v in data["relation_index"].items()
        }
        self.community_index = {
            k: Community(**v) for k, v in data["community_index"].items()
        }
        self.doc_chunk_links = {k: set(v) for k, v in data["doc_chunk_links"].items()}
        self.chunk_entity_links = {
            k: set(v) for k, v in data["chunk_entity_links"].items()
        }
        self.chunk_relation_links = {
            k: set(v) for k, v in data["chunk_relation_links"].items()
        }
        self.entity_chunk_links = {
            k: set(v) for k, v in data["entity_chunk_links"].items()
        }
        self.entity_community_links = {
            k: set(v) for k, v in data["entity_community_links"].items()
        }

    def deduplicate_entities(
        self, entity_groups_to_deduplicate: Dict[tuple[str, str], List[Entity]]
    ) -> None:
        """Deduplicate entities in the docstore.

        This method takes a dictionary of entity groups to deduplicate,
        where each group contains entities that should be merged into one.
        For each group, the first entity is considered the canonical entity,
        and all other entities are merged into it.

        :param entity_groups_to_deduplicate: Dictionary mapping
            (entity_type, normalized_name) tuples to a list of Entity objects
            that should be merged.
        """
        # Create mapping from old entity hashes to new (canonical) entity hashes
        hash_mapping = {}
        new_entity_index = {}

        # First, copy over all entities that aren't being deduplicated
        all_entities_to_dedupe = [
            entity.hash()
            for entities in entity_groups_to_deduplicate.values()
            for entity in entities
        ]
        for entity_hash, entity in self.entity_index.items():
            if entity_hash not in all_entities_to_dedupe:
                new_entity_index[entity_hash] = entity
                hash_mapping[entity_hash] = entity_hash

        # Then handle the deduplication groups
        for entities in entity_groups_to_deduplicate.values():
            canonical_entity = entities[0]  # First entity is canonical
            canonical_hash = canonical_entity.hash()

            # Merge all entities in the group into the canonical entity
            merged_entity = canonical_entity
            for entity in entities[1:]:
                merged_entity = merged_entity + entity
                hash_mapping[entity.hash()] = canonical_hash

            new_entity_index[canonical_hash] = merged_entity
            hash_mapping[canonical_hash] = canonical_hash

        # Update entity index
        self.entity_index = new_entity_index

        # Update relation index to use new entity hashes
        new_relation_index = {}
        for rel_hash, relation in self.relation_index.items():
            source_hash = hash_mapping.get(
                relation.source.hash(), relation.source.hash()
            )
            target_hash = hash_mapping.get(
                relation.target.hash(), relation.target.hash()
            )

            # Create new relation with updated entity references
            new_relation = Relationship(
                source=new_entity_index[source_hash].model_dump(),
                target=new_entity_index[target_hash].model_dump(),
                relation_type=relation.relation_type,
                summary=relation.summary,
            )
            new_relation_index[new_relation.hash()] = new_relation
        self.relation_index = new_relation_index

        # Update chunk-entity links
        new_chunk_entity_links = {}
        for chunk_hash, entity_hashes in self.chunk_entity_links.items():
            new_chunk_entity_links[chunk_hash] = {
                hash_mapping.get(eh, eh) for eh in entity_hashes
            }
        self.chunk_entity_links = new_chunk_entity_links

        # Update chunk-relation links
        new_chunk_relation_links = {}
        for chunk_hash, relation_hashes in self.chunk_relation_links.items():
            new_chunk_relation_links[chunk_hash] = set(relation_hashes)
        self.chunk_relation_links = new_chunk_relation_links

        # Update entity-chunk links
        new_entity_chunk_links = {}
        for entity_hash, chunk_hashes in self.entity_chunk_links.items():
            canonical_hash = hash_mapping.get(entity_hash, entity_hash)
            if canonical_hash not in new_entity_chunk_links:
                new_entity_chunk_links[canonical_hash] = set()
            new_entity_chunk_links[canonical_hash].update(chunk_hashes)
        self.entity_chunk_links = new_entity_chunk_links

        # Update entity-community links
        new_entity_community_links = {}
        for entity_hash, community_hashes in self.entity_community_links.items():
            canonical_hash = hash_mapping.get(entity_hash, entity_hash)
            if canonical_hash not in new_entity_community_links:
                new_entity_community_links[canonical_hash] = set()
            new_entity_community_links[canonical_hash].update(community_hashes)
        self.entity_community_links = new_entity_community_links

        # Save changes
        self._save()


"""Network graph representation of GraphRAG data."""


# from poorman_graphrag.docstore import KnowledgeGraphDocStore


def build_network(
    docstore: "KnowledgeGraphDocStore",
    include_documents: bool = False,
    include_chunks: bool = False,
) -> nx.MultiDiGraph:
    """Build a NetworkX graph from a KnowledgeGraphDocStore.

    The graph will contain nodes for entities and relationships, with optional
    document and chunk nodes. Edges represent the connections between these elements.

    :param docstore: KnowledgeGraphDocStore instance to convert to a network
    :param include_documents: Whether to include document nodes in the graph
    :param include_chunks: Whether to include chunk nodes in the graph
    :return: NetworkX MultiDiGraph containing the network representation
    """
    G = nx.MultiDiGraph()

    # Add document nodes if requested
    if include_documents:
        for doc_hash, doc_text in docstore.doc_index.items():
            G.add_node(doc_hash, type="document", text=doc_text)

    # Add chunk nodes and connect to documents if requested
    if include_chunks:
        for chunk_hash, chunk_text in docstore.chunk_index.items():
            G.add_node(chunk_hash, type="chunk", text=chunk_text)

            # Find parent document and add edge if documents are included
            if include_documents:
                for doc_hash, chunk_hashes in docstore.doc_chunk_links.items():
                    if chunk_hash in chunk_hashes:
                        G.add_edge(doc_hash, chunk_hash, type="contains")

    # Add entity nodes
    for entity_hash, entity in docstore.entity_index.items():
        G.add_node(
            entity_hash,
            type="entity",
            entity_type=entity.entity_type,
            name=entity.name,
            summary=entity.summary,
        )

        # Connect entities to their chunks if chunks are included
        if include_chunks:
            for chunk_hash, entity_hashes in docstore.chunk_entity_links.items():
                if entity_hash in entity_hashes:
                    G.add_edge(chunk_hash, entity_hash, type="mentions")

    # Add relationship nodes and edges
    for rel_hash, relation in docstore.relation_index.items():
        # Get source and target entity hashes
        source_hash = relation.source.hash()
        target_hash = relation.target.hash()

        # Add the relationship edge between entities
        G.add_edge(
            source_hash,
            target_hash,
            key=rel_hash,  # Use rel_hash as edge key for multigraph
            type="relationship",
            relation_type=relation.relation_type,
            summary=relation.summary,
        )

        # Connect relationships to chunks if chunks are included
        if include_chunks:
            for chunk_hash, rel_hashes in docstore.chunk_relation_links.items():
                if rel_hash in rel_hashes:
                    # Add edge from chunk to both entities involved in relationship
                    G.add_edge(chunk_hash, source_hash, type="mentions")
                    G.add_edge(chunk_hash, target_hash, type="mentions")

    return G


# def get_entity_subgraph(
#     G: nx.MultiDiGraph,
#     entity_hash: str,
#     n_hops: int = 2,
#     edge_types: Optional[Set[str]] = None,
# ) -> nx.MultiDiGraph:
#     """Extract a subgraph centered on a specific entity.

#     :param G: Full network graph
#     :param entity_hash: Hash of the entity to center the subgraph on
#     :param n_hops: Number of hops (edge traversals) to include in subgraph
#     :param edge_types: Set of edge types to traverse. If None, traverse all edges.
#     :return: Subgraph centered on the specified entity
#     """
#     if edge_types is None:
#         edge_types = {"mentions", "relationship", "contains"}

#     # Get nodes within n_hops of entity
#     nodes = {entity_hash}
#     current_nodes = {entity_hash}

#     for _ in range(n_hops):
#         next_nodes = set()
#         for node in current_nodes:
#             # Get neighbors through specified edge types
#             for _, nbr, edge_data in G.edges(node, data=True):
#                 if edge_data["type"] in edge_types:
#                     next_nodes.add(nbr)
#             for nbr, _, edge_data in G.in_edges(node, data=True):
#                 if edge_data["type"] in edge_types:
#                     next_nodes.add(nbr)
#         current_nodes = next_nodes - nodes
#         nodes.update(next_nodes)

#     # Return induced subgraph on selected nodes
#     return G.subgraph(nodes).copy()


# def get_chunk_context(
#     G: nx.MultiDiGraph, entity_hash: str, n_hops: int = 2
# ) -> Dict[str, str]:
#     """Get the text of chunks that provide context for an entity.

#     :param G: Network graph
#     :param entity_hash: Hash of entity to get context for
#     :param n_hops: Number of hops to traverse when finding related chunks
#     :return: Dictionary mapping chunk hashes to chunk text
#     """
#     # Get subgraph around entity
#     subgraph = get_entity_subgraph(
#         G, entity_hash, n_hops=n_hops, edge_types={"mentions", "contains"}
#     )

#     # Find chunks in subgraph
#     chunks = {}
#     for node in subgraph.nodes():
#         node_data = subgraph.nodes[node]
#         if node_data.get("type") == "chunk":
#             chunks[node] = node_data["text"]

#     return chunks
