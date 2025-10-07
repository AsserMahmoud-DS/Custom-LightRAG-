# Custom LightRAG

This project implements a lightweight Retrieval-Augmented Generation (RAG) pipeline using LlamaIndex, Pinecone, and Groq LLMs. It extracts knowledge from PDFs, builds a knowledge graph, and enables hybrid semantic/document querying.

---

## Project Structure

```
Custom-LightRAG/
│
├── data/                # PDF source documents for ingestion
├── graph_storage/       # Stores the serialized knowledge graph (kg_graph.pkl)
├── rag/                 # Core RAG pipeline modules
│   ├── __init__.py
│   ├── entity_extractor.py
│   ├── kg_store.py
│   ├── loaders.py
│   ├── query_engine.py
│   ├── retriever.py
│   ├── utils.py
│   ├── vector_index.py
│   └── __pycache__/
└── main.py              # Entry point for running the pipeline
```

---

## File Overview

- **main.py**  
  Orchestrates the pipeline: loads documents, chunks them, builds vector indexes, extracts entities/relations, updates the knowledge graph, and runs the query loop.

- **rag/loaders.py**  
  Loads PDF documents from the `data/` folder and splits them into text chunks.

- **rag/vector_index.py**  
  Builds a Pinecone vector index for document chunks, using deterministic IDs for deduplication.

- **rag/entity_extractor.py**  
  Uses an LLM to extract entities and relations from each chunk. Deduplicates and upserts entities/relations into Pinecone, tracking provenance via chunk IDs.

- **rag/kg_store.py**  
  Manages the in-memory knowledge graph (NetworkX). Adds entities and relations, supports neighbor and multi-hop queries, and saves/loads the graph.

- **rag/retriever.py**  
  Builds hybrid retrievers that combine chunk, entity, and relation retrieval for better RAG performance.

- **rag/query_engine.py**  
  Wraps the retriever with a query engine that synthesizes answers using the LLM.

- **rag/utils.py**  
  Utility functions, including deterministic ID generation (`make_id`) for chunks, entities, and relations.

---

## Pipeline Flow (main.py)

```python
# 1. Initialize models (Groq LLM, HuggingFace embedding)
# 2. Load and chunk documents from ./data
# 3. Build Pinecone vector index for chunks
# 4. For each chunk:
#    - Check if already processed (by chunk_id)
#    - If new, extract entities/relations and upsert to Pinecone
#    - Update the knowledge graph
# 5. Save the knowledge graph
# 6. Build hybrid retrievers (chunks, entities, relations, KG)
# 7. Build query engine and run interactive queries
```

---

## Usage

1. **Place your PDF files in the `data/` folder.**
2. **Set your API keys as environment variables:**
   - `GROQ_API_KEY`
   - `PINECONE_API_KEY`
3. **Run the pipeline:**
   ```bash
   python main.py
   ```
4. **Ask questions interactively!**

---

## Example Query

```
Ask a question (or 'exit'): who is asser mahmoud
Answer: [LLM-generated response based on your documents and KG]
```

---

## Notes

- The pipeline avoids duplicate chunk/entity/relation upserts using deterministic IDs and provenance tracking.
- The knowledge graph is persisted in `graph_storage/kg_graph.pkl`.
- For more details, see comments in each file.

---
