import os.path
import pickle

import torch
from transformers import BertLMHeadModel, BertTokenizerFast
from util import *
from initial_rank import *

model = BertLMHeadModel.from_pretrained("ielab/TILDE")
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


def perform_re_rank(query, sims, embeddings, stop_ids):
    query_token_ids = tokenizer(query, add_special_tokens=False)["input_ids"]
    cleaned_query_token_ids = [id for id in query_token_ids if id not in stop_ids]  # only keep valid token ids

    query_inputs = tokenizer([query], return_tensors="pt", padding=True, truncation=True)
    query_input_ids = query_inputs["input_ids"]
    query_input_ids[:, 0] = 2  # 2 is the token id for [QRY]
    with torch.no_grad():
        query_outputs = model(input_ids=query_input_ids,
                              token_type_ids=query_inputs["token_type_ids"],
                              attention_mask=query_inputs["attention_mask"],
                              return_dict=True).logits[:, 0]
    query_probs = torch.sigmoid(query_outputs)
    query_log_probs = torch.log10(query_probs)[0].cpu().numpy()

    QL_scores = []
    DL_scores = []


def main():
    if not os.path.exists("tilde_embeddings.pkl"):
        raise FileNotFoundError("tilde_embeddings.pkl not found, make sure to generate the tilde embeddings first")
    with open('tilde_embeddings.pkl', 'rb') as handle:
        doc_embeddings = pickle.load(handle)

    corpus, queries, train_rel = prepare_data()
    dictionary, lsi, index = create_model(corpus)
    stop_ids = get_stop_ids(tokenizer)
    rr_basic, rr_reranked = [], []
    for query in queries:
        sims = run_first_stage_retrieval(query, dictionary, lsi, index)
        rr1 = reciprocal_rank(sims, train_rel, query)
        rerank = perform_re_rank(query, sims, doc_embeddings, stop_ids)
        rr2 = reciprocal_rank(rerank, train_rel, query)
        print(f"[Query {query['_id']}] RR Basic: {rr1}, RR Reranked: {rr2}")
        rr_basic.append(rr1)
        rr_reranked.append(rr2)
    print("-" * 50)
    print(f"MRR Basic: {sum(rr_basic) / len(queries)}")
    print(f"MRR Reranked: {sum(rr_reranked) / len(queries)}")


if __name__ == '__main__':
    main()
