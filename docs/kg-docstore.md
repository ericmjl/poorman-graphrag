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

Supports multiple retrieval methods:

- Direct hash lookups for all node types
- Keyword search using BM25 ranking
- Filtering by:
  - Chunks
  - Documents
  - Relations
  - Entities
  - Communities
- Configurable number of results

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
