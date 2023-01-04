import json
import os.path

CORPUS_SUBSET_SIZE = 10_000
DATASET_PATH = "../dataset"

def get_corpus_json():
    subset = []

    if not os.path.exists(f"{DATASET_PATH}/corpus.jsonl"):
        raise FileNotFoundError(f"corpus.jsonl not found, make sure to put the dataset in {DATASET_PATH}")

    with open(f"{DATASET_PATH}/corpus.jsonl", "r") as f:
        for i in range(CORPUS_SUBSET_SIZE):
            line = f.readline()
            subset.append(json.loads(line))
    return subset


def get_train_json():
    subset_train_rel = []

    if not os.path.exists(f"{DATASET_PATH}/qrels/train.tsv"):
        raise FileNotFoundError(f"train.tsv not found, make sure to put the dataset in {DATASET_PATH}")

    with open(f"{DATASET_PATH}/qrels/train.tsv", "r") as f:
        f.readline()
        while True:
            query_id, corpus_id, rel = f.readline().split("\t")
            if int(corpus_id) >= CORPUS_SUBSET_SIZE:
                break
            subset_train_rel.append({
                "query_id": int(query_id),
                "corpus_id": int(corpus_id),
                "rel": int(rel)
            })
    return subset_train_rel


def get_queries_json(subset_train_rel):
    queries_subset = []
    query_ids = [item["query_id"] for item in subset_train_rel]

    if not os.path.exists(f"{DATASET_PATH}/queries.jsonl"):
        raise FileNotFoundError(f"queries.jsonl not found, make sure to put the dataset in {DATASET_PATH}")

    with open(f"{DATASET_PATH}/queries.jsonl") as f:
        while True:
            line = f.readline()
            if not line:
                break
            else:
                line = json.loads(line)
            if int(line["_id"]) in query_ids:
                queries_subset.append(line)
            if not query_ids:
                break
    return queries_subset


if __name__ == '__main__':
    subset = get_corpus_json()
    subset_train_rel = get_train_json()
    queries_subset = get_queries_json(subset_train_rel)
    with open("corpus_subset.json", "w") as f:
        f.write(json.dumps(subset, indent=4))
    with open("qrel_subset.json", "w") as f:
        f.write(json.dumps(subset_train_rel, indent=4))
    with open("queries_subset.json", "w") as f:
        f.write(json.dumps(queries_subset, indent=4))



