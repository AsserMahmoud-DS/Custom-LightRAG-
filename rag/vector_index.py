# rag/vector_index.py
import os
from rag.utils import make_id
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from pinecone import Pinecone, ServerlessSpec


def build_vector_index(nodes, index_name="lightrag-chunks"):
    """
    Incrementally upsert document chunks into Pinecone.
    Returns a VectorStoreIndex (not raw Pinecone Index),
    works with VectorIndexRetriever.
    Stores chunk text under metadata["_node_content"] as required by LlamaIndex.
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # ensure index exists
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    # Wrap Pinecone index for LlamaIndex
    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # build docs with deterministic IDs
    docs = []
    for node in nodes:
        if not node.text:
            continue
        meta = {
            "text": node.text,   # store text for retrieval

        }
        node_doc_id = getattr(node, "doc_id", None)
        if node_doc_id is not None:
            meta["doc_id"] = str(node_doc_id)
        docs.append({"id": make_id(node.text), "metadata": meta})  


    if docs:
        # check which IDs already exist
        ids = [d["id"] for d in docs]
        resp = pinecone_index.fetch(ids=ids)
        if isinstance(resp, dict):
            existing = resp.get("vectors", {})
        else:
            existing = getattr(resp, "vectors", {}) or {}

        new_docs = [d for d in docs if d["id"] not in existing]

        if new_docs:
            texts = [d["metadata"]["text"] for d in new_docs]  # pull text from metadata
            embeddings = embed_model.get_text_embedding_batch(texts)
            vectors = [
                {"id": d["id"], "values": emb, "metadata": d["metadata"]}
                for d, emb in zip(new_docs, embeddings)
            ]

            pinecone_index.upsert(vectors=vectors)
            print(f"Upserted {len(vectors)} new chunks.")
        else:
            print("No new chunks to upsert.")

    # Return a real LlamaIndex VectorStoreIndex
    return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
