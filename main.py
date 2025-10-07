# main.py
from llama_index.llms.groq import Groq
import os
from rag.loaders import load_docs
from rag.vector_index import build_vector_index
from rag.kg_store import KGStore
from rag.retriever import build_light_rag_retriever
from rag.query_engine import build_query_engine
from rag.utils import make_id
from llama_index.core import Settings
from rag.entity_extractor import extract_entities_and_relations, store_entities_and_relations
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
from pinecone import Pinecone
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def load_entities_relations_indices():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    entity_store = PineconeVectorStore(pinecone_index=pc.Index("lightrag-entities"))
    relation_store = PineconeVectorStore(pinecone_index=pc.Index("lightrag-relations"))

    entities_index = VectorStoreIndex.from_vector_store(entity_store)
    relations_index = VectorStoreIndex.from_vector_store(relation_store)

    return entities_index, relations_index


def chunk_already_processed(entity_index, chunk_id: str) -> bool:
    """
    Check Pinecone if any entity already has this chunk_id in its metadata.
    """
    try:
        results = entity_index.query(
            vector=[0.0] * 384,   # dummy vector, only filter is used
            top_k=1,
            filter={"chunk_ids": {"$in": [chunk_id]}},
        )
        return bool(results and getattr(results, "matches", []))
    except Exception as e:
        print(f"[warn] Could not check chunk {chunk_id}: {e}")
        return False


def main():
    # 1. Init models
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")
    llm = Groq(model="openai/gpt-oss-20b", api_key=groq_api_key)
    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 2. Load & chunk documents
    nodes = load_docs("./data")

    # 3. Build Pinecone index for chunks
    chunks_index = build_vector_index(nodes)

    # 4. Extract entities/relations only for NEW chunks
    kg_store = KGStore()
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    entity_index = pc.Index("lightrag-entities")

    for node in nodes:
        chunk_id = make_id(node.text)

        if chunk_already_processed(entity_index, chunk_id):
            print(f"[skip] Chunk {chunk_id} already processed.")
            continue

        # new chunk → run extraction + store
        data = extract_entities_and_relations(llm, node.text)
        entities, relations = data["entities"], data["relations"]

        store_entities_and_relations(entities, relations, chunk_id=chunk_id)
        kg_store.add_entities_relations(entities, relations)

    kg_store.save()

    # 5. Hybrid retrievers
    entities_index, relations_index = load_entities_relations_indices()
    retriever = build_light_rag_retriever(
        chunks_index, entities_index, relations_index, kg_store, llm, mode="auto"
    )

    # 6. Query engine
    query_engine = build_query_engine(retriever, llm)

    # 7. Run queries
    while True:
        q = input("\nAsk a question (or 'exit'): ")
        if q.lower() == "exit":
            break
        response = query_engine.query(q)
        print("Answer:", response)


if __name__ == "__main__":
    main()
