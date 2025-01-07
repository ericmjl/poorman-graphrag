# KnowledgeGraphDocstore Design Document

## Overview

The KnowledgeGraphDocstore is an implementation of LlamaBot's AbstractDocumentStore that uses a knowledge graph structure to store and manage documents, chunks, entities, relationships, and communities. It uses NetworkX's MultiDiGraph as its core data structure.

## Core Design Choices

### Graph Structure

The knowledge graph uses a partitioned structure with multiple types of nodes:

1. Document nodes
   - Represent full documents added to the store
   - Identified by SHA256 hash of content
   - Store full text as node attribute
   - Partition type: "document"

2. Chunk nodes
   - Represent segments of documents
   - Connected to parent documents via "document_chunk" edges
   - Identified by SHA256 hash of content
   - Store chunk text as node attribute
   - Partition type: "chunk"

3. Entity nodes
   - Represent entities extracted from chunks
   - Connected to source chunks via "chunk_entity" edges
   - Store Entity Pydantic models as node attributes
   - Support entity deduplication and merging
   - Partition type: "entity"

### Edge Types

1. Document → Chunk edges
   - Partition type: "document_chunk"
   - Connect documents to their constituent chunks

2. Chunk → Entity edges
   - Partition type: "chunk_entity"
   - Connect text chunks to extracted entities

3. Entity → Entity edges
   - Store relationships between entities
   - Include Relationship Pydantic models as edge attributes
   - Track source chunks in "chunk_hashes" attribute

### Community Detection

- Uses NetworkX's implementation of Louvain community detection algorithm
- Communities are stored separately in a communities dictionary
- Each community has:
  - Set of member entities (Entities model)
  - AI-generated summary
  - Unique hash identifier
- Community membership is tracked on nodes via "community" attribute

### Data Processing Pipeline

When adding documents (append method):

1. Document Addition
   - Hash document text using SHA256
   - Store as document node

2. Chunking
   - Uses SDPMChunker for document segmentation
   - Creates chunk nodes linked to document
   - Uses "document_chunk" edges for linkage

3. Entity Extraction
   - Uses StructuredBot with custom prompts to extract entities
   - Creates Entity nodes from extracted entities
   - Links entities to source chunks

4. Relationship Extraction
   - Uses StructuredBot to identify relationships between entities
   - Creates relationship edges with Relationship models
   - Tracks source chunks for relationships

5. Entity Deduplication
   - Uses Levenshtein distance to identify similar entities
   - StructuredBot judges if entities should be merged
   - Merges entities by combining their attributes
   - Rewires graph connections to canonical entities

6. Community Detection
   - Creates entity subgraph
   - Runs Louvain algorithm
   - Generates community summaries using StructuredBot
   - Stores communities with member entities and summaries

### Persistence

- JSON-based storage format
- Stores:
  - Node attributes and partition types
  - Edge attributes and relationships
  - Community data
  - Custom serialization for Pydantic models
- Automatic saving after modifications
- Creates parent directories if needed

### LLM Integration

Uses llamabot StructuredBot instances for:

1. Entity Extraction
   - Extracts structured Entity objects from chunks
   - Configurable prompts

2. Relationship Extraction
   - Identifies relationships between entities
   - Returns structured Relationship objects

3. Entity Similarity Judging
   - Determines if similar entities should be merged
   - Uses custom prompts for comparison

4. Community Summarization
   - Generates summaries for detected communities
   - Analyzes community subgraphs

### Retrieval Capabilities

The KnowledgeGraphDocstore supports specialized retrieval methods focused on graph-theoretic and entity-based queries:

1. Entity-Centric Retrieval (Implemented)
   - Uses BM25 ranking for multi-word entity name matching
   - Entity results are formatted as "entity_name: summary"
   - Entity summaries are pre-generated during entity extraction
   - Summaries are stored as a list of strings and joined during retrieval

2. Community-Based Retrieval (Implemented)
   - Direct substring matching on community summaries
   - Community summaries are pre-generated during community detection
   - Returns full community summaries as-is

3. Relationship-Based Retrieval (Tentative - Not Yet Implemented)
   - Find all relationships between given entities
   - Discover entities with specific relationship patterns
   - Relationship path exploration with type constraints
   - Subgraph extraction based on relationship types
   - Relationship strength scoring based on frequency

4. Graph Pattern Matching (Tentative - Not Yet Implemented)
   - Exact subgraph pattern matching
   - Approximate pattern matching with similarity thresholds
   - Motif discovery and matching
   - Template-based graph querying

Key Design Decisions:
1. Single Query Interface
   - One query string searches both entities and communities
   - Simplifies the API to match other docstore implementations
   - No need for separate entity/community search parameters

2. BM25 for Entity Search
   - Chosen over Levenshtein distance for better multi-word matching
   - Handles partial matches and word importance weighting
   - No arbitrary similarity threshold needed
   - Reuses existing dependency from LlamaBot

3. Pre-generated Summaries
   - Both entities and communities have pre-generated summaries
   - No need for runtime summary generation during retrieval
   - Consistent summary format across all retrievals
   - Entity summaries are stored as lists for flexibility but joined for display

4. Result Formatting
   - Entity results prefixed with name for clear attribution
   - Community summaries kept as-is since they're self-contained
   - Consistent with other docstore implementations while preserving graph context

Each retrieval method assumes specific keywords or identifiers are provided:
- Entity retrieval requires exact entity names or types
- Relationship queries need relationship types or entity endpoints
- Community retrieval uses community IDs or member entity names
- Pattern matching expects well-defined graph patterns

The retrieval system focuses on graph structure and metadata rather than raw text content, enabling efficient navigation and discovery within the knowledge graph.

### Design Philosophy

1. Type Safety
   - Uses Pydantic models for structured data
   - Type hints throughout codebase

2. Graph-Centric Architecture
   - NetworkX MultiDiGraph for flexibility
   - Partition-based node organization
   - Edge attributes for relationships

3. LLM Integration
   - Modular StructuredBot usage
   - Configurable prompts
   - Separation of LLM tasks

4. Persistence
   - JSON-based storage
   - Custom serialization
   - Automatic saving

5. Extensibility
   - Configurable LLM components
   - Customizable prompts
   - Modular processing pipeline
