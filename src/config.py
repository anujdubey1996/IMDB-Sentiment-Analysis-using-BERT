import os
import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer

DATA_COLUMN = "review"
LABEL_COLUMN = "sentiment"

RANDOM_SEED = 42

bert_model_name="uncased_L-12_H-768_A-12"

bert_ckpt_dir = os.path.join("../model/", bert_model_name)
bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")

tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))