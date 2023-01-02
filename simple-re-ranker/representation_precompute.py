import pickle

import numpy as np
from sentence_transformers import SentenceTransformer
from util import *

rerank_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def pre_compute_embeddings(corpus):
    print("Precomputing embeddings for re ranking")
    pre_computed = []
    for idx, item in enumerate(corpus):
        progressBar(idx, len(corpus))
        doc_embedding = np.array(rerank_model.encode(item["text"]))
        pre_computed.append(doc_embedding)
    with open('representation_embeddings.pkl', 'wb') as f:
        pickle.dump(pre_computed, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    corpus, queries, train_rel = prepare_data()
    pre_compute_embeddings(corpus)
