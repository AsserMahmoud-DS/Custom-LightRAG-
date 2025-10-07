from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

def load_docs(path="./data"):
    docs = SimpleDirectoryReader(path).load_data()
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(docs)
    return nodes
