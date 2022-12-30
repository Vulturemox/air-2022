import json

CORPUS_SUBSET_SIZE = 10_000

subset = []
with open("../dataset/corpus.jsonl", "r") as f:
    for i in range(CORPUS_SUBSET_SIZE):
        line = f.readline()
        subset.append(json.loads(line))
with open("corpus_subset.json", "w") as f:
    f.write(json.dumps(subset, indent=4))

subset_train_rel = []

with open("../dataset/qrels/train.tsv", "r") as f:
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
with open("qrel_subset.json", "w") as f:
    f.write(json.dumps(subset_train_rel, indent=4))

queries_subset = []
query_ids = [item["query_id"] for item in subset_train_rel]
init_len = len(query_ids)
with open("../dataset/queries.jsonl") as f:
    while True:
        line = f.readline()
        if not line:
            break
        else:
            line = json.loads(line)
        if int(line["_id"]) in query_ids:
            queries_subset.append(line)
            print(f"{len(queries_subset)} / {init_len}")
        if not query_ids:
            break
with open("queries_subset.json", "w") as f:
    f.write(json.dumps(queries_subset, indent=4))



