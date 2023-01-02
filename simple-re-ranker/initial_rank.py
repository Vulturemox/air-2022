from gensim import corpora, models, similarities
from gensim.parsing.preprocessing import preprocess_string

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
