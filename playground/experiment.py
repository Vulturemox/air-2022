import json

import numpy as np
from gensim import corpora, models, similarities
from gensim.parsing.preprocessing import preprocess_string
from sentence_transformers import SentenceTransformer

corpus = []
corpus_raw = []
print("Preprocessing")
with open("../dataset/subset/corpus_subset.json", "r") as f:
    lines = json.load(f)
    for item in lines:
        corpus.append(preprocess_string(item["text"]))
        corpus_raw.append(item["text"])

queries = []
queries_raw = []
with open("../dataset/subset/queries_subset.json", "r") as f:
    lines = json.load(f)
    for item in lines:
        queries.append(preprocess_string(item["text"]))
        queries_raw.append(item["text"])

print("Creating BOW")
dictionary = corpora.Dictionary(corpus)
bow_corpus = [dictionary.doc2bow(text) for text in corpus]
print("Creating LSI Model")
lsi = models.LsiModel(bow_corpus, id2word=dictionary, num_topics=400)
index = similarities.MatrixSimilarity(lsi[bow_corpus])

print("Running test query")
vec_bow = dictionary.doc2bow(queries[0])
vec_lsi = lsi[vec_bow]  # convert the query to LSI space
sims = index[vec_lsi]  # perform a similarity query against the corpus
sims = sorted(enumerate(sims), key=lambda item: -item[1])
print("Query: ", queries_raw[0])
for doc_position, doc_score in sims[:10]:
    print(f"[{doc_position}]", doc_score, corpus_raw[doc_position])
for i, doc in enumerate(sims):
    if doc[0] == 0:
        print(i)
        break
print(len(sims))

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
query_embedding = np.array(model.encode(queries_raw[0]))
reranked = []
for doc_position, doc_score in sims[:100]:
    doc_embedding = np.array(model.encode(corpus_raw[doc_position]))
    cos_sim = (doc_embedding.T @ query_embedding) / (np.linalg.norm(doc_embedding) * np.linalg.norm(query_embedding))
    reranked.append((cos_sim, doc_position))
reranked = sorted(reranked, key=lambda item: -item[0])
for item in reranked[:10]:
    print(f"[{item[1]}]", item[0], corpus_raw[item[1]])