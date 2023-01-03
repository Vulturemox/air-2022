import pickle

import torch
from transformers import BertLMHeadModel, BertTokenizerFast
import numpy as np
import extract_sub_dataset as dataset
from util import get_stop_ids, progressBar

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = BertLMHeadModel.from_pretrained("ielab/TILDE")
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# SOURCE: https://github.com/ielab/TILDE/blob/main/indexing.py
# Repo for the Paper "TILDE: Term Independent Likelihood moDEl for Passage Re-ranking"
# Authors: Shengyao Zhuang and Guido Zuccon
#
# Code was adapted to fit our needs
def get_embedding(sample_text, stop_ids):
    passage_token_ids = tokenizer(sample_text, add_special_tokens=False)["input_ids"]
    cleaned_ids = np.array([id for id in passage_token_ids if id not in stop_ids]).astype(np.int16)
    passage_inputs = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True)
    passage_input_ids = passage_inputs["input_ids"]
    passage_input_ids[:, 0] = 1  # 1 is the token id for [DOC]
    with torch.no_grad():
        passage_outputs = model(input_ids=passage_input_ids,
                                token_type_ids=passage_inputs["token_type_ids"],
                                attention_mask=passage_inputs["attention_mask"],
                                return_dict=True).logits[:, 0]
        passage_probs = torch.sigmoid(passage_outputs)
        passage_log_probs = torch.squeeze(torch.log10(passage_probs)).cpu().numpy().astype(np.float16)
    return passage_log_probs, cleaned_ids

def tilde_pre_compute():
    corpus = dataset.get_corpus_json()
    stopwords = get_stop_ids(tokenizer)
    pre_computed = []
    for idx, item in enumerate(corpus):
        progressBar(idx, len(corpus))
        doc_embedding = get_embedding(item["text"], stopwords)
        pre_computed.append(doc_embedding)
    with open('passage_embeddings.pkl', 'wb') as f:
        pickle.dump(pre_computed, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    tilde_pre_compute()
