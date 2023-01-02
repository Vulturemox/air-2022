import os
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer
from util import *
from initial_rank import *

rerank_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def perform_re_rank(query, sims, embeddings):
    query_embedding = np.array(rerank_model.encode(query["text"]))
    reranked = []
    for doc_position, doc_score in sims:
        doc_embedding = embeddings[doc_position]
        cos_sim = (doc_embedding.T @ query_embedding) / (np.linalg.norm(doc_embedding) * np.linalg.norm(query_embedding))
        reranked.append((doc_position, cos_sim))
    reranked = sorted(reranked, key=lambda item: -item[1])
    return reranked

def main():
    if not os.path.exists("representation_embeddings.pkl"):
        raise FileNotFoundError("representation_embeddingss.pkl not found, make sure to generate the representation embeddings first")
    with open('representation_embeddings.pkl', 'rb') as handle:
        embeddings = pickle.load(handle)

    corpus, queries, train_rel = prepare_data()
    dictionary, lsi, index = create_model(corpus)
    rr_basic, rr_reranked = [], []
    for query in queries:
        sims = run_first_stage_retrieval(query, dictionary, lsi, index)
        rr1 = reciprocal_rank(sims, train_rel, query)
        rerank = perform_re_rank(query, sims, embeddings)
        rr2 = reciprocal_rank(rerank, train_rel, query)
        print(f"[Query {query['_id']}] RR Basic: {rr1}, RR Reranked: {rr2}")
        rr_basic.append(rr1)
        rr_reranked.append(rr2)
    print("-" * 50)
    print(f"MRR Basic: {sum(rr_basic) / len(queries)}")
    print(f"MRR Reranked: {sum(rr_reranked) / len(queries)}")


if __name__ == '__main__':
    main()
