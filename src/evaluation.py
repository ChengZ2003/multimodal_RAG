import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_path)

from mmRAG.utils.data_preprocessing import load_dataset, parse_pdf, convert_to_documents
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext
)
from llama_index.postprocessor.flag_embedding_reranker import (
    FlagEmbeddingReranker,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from ragas.integrations.llama_index import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# def ragas_evaluation(
#     query_engine,
#     metrics,
#     dataset,
#     llm,
#     embeddings,
#     raise_exceptions
# ):
#     result = evaluate(query_engine,
#                       metrics,
#                       dataset,
#                       llm,
#                       embeddings,
#                       raise_exceptions)
#     return result.to_pandas()

def beir_evaluation():
    pass


if __name__ == '__main__':
    llm = Ollama(model="qwen2", request_timeout=60.0)
    embed_path = "/home/project/data/jc/mmRAG/model/bge-m3"
    embed_model = HuggingFaceEmbedding(embed_path)

    Settings.llm = llm
    Settings.embed_model = embed_model

    vector_store = MilvusVectorStore(
        uri="http://localhost:19530/",
        token="root:Milvus",
        collection_name='doc_3900889',
        dim=1024,
        overwrite=True,
        enable_sparse=True,
        hybrid_ranker="RRFRanker",
        hybrid_ranker_params={"k": 60},
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    document_path = '/home/project/data/jc/mmRAG/evaluation/3900889.pdf'

    raw_docs = parse_pdf(document_path)
    documents = convert_to_documents(raw_docs)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )

    rerank = FlagEmbeddingReranker(model="BAAI/bge-reranker-large", top_n=5)

    query_engine = index.as_query_engine(
        similarity_top_k=10, node_postprocessors=[rerank]
    )

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    dataset = load_dataset('/home/project/data/jc/mmRAG/evaluation/data/3900889/testset.csv')

    # print(type(dataset))

    result = evaluate(
        query_engine = query_engine,
        metrics = metrics,
        dataset = dataset,
        llm = llm,
        embeddings = embed_model,
        raise_exceptions = False
    )

    result.to_pandas().to_csv('result.csv')