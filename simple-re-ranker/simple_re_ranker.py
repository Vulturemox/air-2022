import json
import sys

import numpy as np
from gensim import corpora, models, similarities
from gensim.parsing.preprocessing import preprocess_string
from sentence_transformers import SentenceTransformer

import extract_sub_dataset as dataset

rerank_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def progressBar(value, endvalue, bar_length=20):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write(f"\rPercent: [{arrow + spaces}] {int(round(percent * 100))}% {value} of {endvalue}")
    sys.stdout.flush()

def prepare_data():
    corpus_subset = dataset.get_corpus_json()
    subset_train_rel = dataset.get_train_json()
    queries_subset = dataset.get_queries_json(subset_train_rel)
    return corpus_subset, queries_subset, subset_train_rel

def pre_compute_embeddings(corpus):
    print("Precomputing embeddings for re ranking")
    pre_computed = []
    for idx, item in enumerate(corpus):
        progressBar(idx, len(corpus))
        doc_embedding = np.array(rerank_model.encode(item["text"]))
        pre_computed.append(doc_embedding)
    print("\n")
    return pre_computed

def create_model(corpus):
    print("Creating BOW")
    pre_processed = [preprocess_string(item["text"]) for item in corpus]
    dictionary = corpora.Dictionary(pre_processed)
    bow_corpus = [dictionary.doc2bow(text) for text in pre_processed]
    print("Creating LSI Model")
    lsi = models.LsiModel(bow_corpus, id2word=dictionary, num_topics=400)
    index = similarities.MatrixSimilarity(lsi[bow_corpus])
    return dictionary, lsi, index

def run_first_stage_retrieval(query, dictionary, lsi, index):
    vec_bow = dictionary.doc2bow(preprocess_string(query["text"]))
    vec_lsi = lsi[vec_bow]  # convert the query to LSI space
    sims = index[vec_lsi]  # perform a similarity query against the corpus
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    return sims[:1000]

def perform_re_rank(query, sims, embeddings):
    query_embedding = np.array(rerank_model.encode(query["text"]))
    reranked = []
    for doc_position, doc_score in sims:
        doc_embedding = embeddings[doc_position]
        cos_sim = (doc_embedding.T @ query_embedding) / (np.linalg.norm(doc_embedding) * np.linalg.norm(query_embedding))
        reranked.append((doc_position, cos_sim))
    reranked = sorted(reranked, key=lambda item: -item[1])
    return reranked

def reciprocal_rank(sims, rel, query):
    rel_item = [item for item in rel if item["query_id"] == int(query["_id"])][0]
    for i, doc in enumerate(sims):
        if doc[0] == rel_item["corpus_id"]:
            return 1 / (i + 1)
    return 0


def main():
    corpus, queries, train_rel = prepare_data()
    embeddings = pre_compute_embeddings(corpus)
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
