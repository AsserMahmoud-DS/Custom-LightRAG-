from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer

def build_query_engine(retriever, llm):
    response_synth = get_response_synthesizer(llm=llm)
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        response_synthesizer=response_synth
    )
    return query_engine
