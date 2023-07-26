import os
import torch
from torch.nn import functional as F
import string
from transformers import BertTokenizer, BertForMaskedLM, top_k_top_p_filtering, logging
logging.set_verbosity_error()

no_words_to_be_predicted = globals()
select_model = globals()
enter_input_text = globals()

def set_model_config(**kwargs):
  for key, value in kwargs.items():
    print("{0} = {1}".format(key, value))
  
  no_words_to_be_predicted = list(kwargs.values())[0] # integer values
  select_model = list(kwargs.values())[1] # possible values = 'bert' or 'gpt' or 'xlnet'
  enter_input_text = list(kwargs.values())[2] #only string

  return no_words_to_be_predicted, select_model, enter_input_text


def load_model(model_name):
  try:
    if model_name.lower() == "bert":
      bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
      bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()
      return bert_tokenizer,bert_model
    else:
        print("tf?")
  except Exception as e:
    pass


def get_all_predictions(text_sentence,  model_name, top_clean=5):
  if model_name.lower() == "bert":
    # ========================= BERT =================================
    input_ids, mask_idx = encode_bert(bert_tokenizer, text_sentence)
    with torch.no_grad():
      predict = bert_model(input_ids)[0]
    bert = decode_bert(bert_tokenizer, predict[0, mask_idx, :].topk(no_words_to_be_predicted).indices.tolist(), top_clean)
    return {'bert': bert}


# bert encode
def encode_bert(tokenizer, text_sentence, add_special_tokens=True):
  text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
  # if <mask> is the last token, append a "." so that models dont predict punctuation.
  if tokenizer.mask_token == text_sentence.split()[-1]:
    text_sentence += ' .'
    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
  return input_ids, mask_idx
  
# bert decode
def decode_bert(tokenizer, pred_idx, top_clean):
  ignore_tokens = string.punctuation + '[PAD]'
  tokens = []
  for w in pred_idx:
    token = ''.join(tokenizer.decode(w).split())
    if token not in ignore_tokens:
      tokens.append(token.replace('##', ''))
  return '\n'.join(tokens[:top_clean])


def get_prediction_end_of_sentence(input_text, model_name):
  try:
    if model_name.lower() == "bert":
      input_text += ' <mask>'
      print(input_text)
      res = get_all_predictions(input_text, model_name, top_clean=int(no_words_to_be_predicted)) 
      return res
    else:
        print("Tf2?")

  except Exception as error:
    pass

try:
  print("Next Word Prediction with Pytorch using BERT")
  no_words_to_be_predicted, select_model, enter_input_text = set_model_config(no_words_to_be_predicted=5, select_model = "bert", enter_input_text = "How are you gonna cook the")
  if select_model.lower() == "bert":
    bert_tokenizer, bert_model  = load_model(select_model)
    res = get_prediction_end_of_sentence(enter_input_text, select_model)
    # print("result is: {}" .format(res))
    answer_bert = []
    print(res['bert'].split("\n"))
    for i in res['bert'].split("\n"):
      answer_bert.append(i)
      answer_as_string_bert = "    ".join(answer_bert)
except Exception as e:
  print('Some problem occured')