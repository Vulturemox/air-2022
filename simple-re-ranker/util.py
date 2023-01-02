import re
import sys
import extract_sub_dataset as dataset

from nltk.corpus import stopwords


def progressBar(value, endvalue, bar_length=20):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write(f"\rPercent: [{arrow + spaces}] {int(round(percent * 100))}% {value} of {endvalue}")
    sys.stdout.flush()

def reciprocal_rank(sims, rel, query):
    rel_item = [item for item in rel if item["query_id"] == int(query["_id"])][0]
    for i, doc in enumerate(sims):
        if doc[0] == rel_item["corpus_id"]:
            return 1 / (i + 1)
    return 0

def prepare_data():
    corpus_subset = dataset.get_corpus_json()
    subset_train_rel = dataset.get_train_json()
    queries_subset = dataset.get_queries_json(subset_train_rel)
    return corpus_subset, queries_subset, subset_train_rel

def get_stop_ids(tok):
    stop_words = set(stopwords.words('english'))
    # keep some common words in ms marco questions
    stop_words.difference_update(["where", "how", "what", "when", "which", "why", "who"])
    vocab = tok.get_vocab()
    tokens = vocab.keys()

    stop_ids = []

    for stop_word in stop_words:
        ids = tok(stop_word, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            stop_ids.append(ids[0])

    for token in tokens:
        token_id = vocab[token]
        if token_id in stop_ids:
            continue
        if token == '##s':  # remove 's' suffix
            stop_ids.append(token_id)
        if token[0] == '#' and len(token) > 1:  # skip most of subtokens
            continue
        if not re.match("^[A-Za-z0-9_-]*$", token):  # remove numbers, symbols, etc..
            stop_ids.append(token_id)

    return set(stop_ids)