from transformers import BertModel, BertConfig, BertTokenizer
import torch
from torch import nn
import numpy as np

class BertTextNet(nn.Module):
    def __init__(self, code_length): 
        super(BertTextNet, self).__init__()

        modelConfig = BertConfig.from_pretrained('bert-base-uncased')
        self.textExtractor = BertModel.from_pretrained('bert-base-uncased', config=modelConfig)
        embedding_dim = self.textExtractor.config.hidden_size

        self.fc = nn.Linear(embedding_dim, code_length)
        self.tanh = torch.nn.Tanh()

    def forward(self, tokens, segments, input_masks):
        output = self.textExtractor(tokens, token_type_ids=segments,
                                    attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :]
        # output[0](batch size, sequence length, model hidden dimension)

        features = self.fc(text_embeddings)
        features = self.tanh(features)
        return features


textNet = BertTextNet(code_length=128)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

tokens, segments, input_masks = [], [], []
for text in texts:

    tokenized_text = text.strip(" ").split(' ')
    print(tokenized_text)
    tokenized_text = tokenized_text[:600]
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)  
    tokens.append(indexed_tokens)
    segments.append([0] * len(indexed_tokens))
    input_masks.append([1] * len(indexed_tokens))

max_len = max([len(single) for single in tokens]) 

for j in range(len(tokens)):
    padding = [0] * (max_len - len(tokens[j]))
    tokens[j] += padding
    segments[j] += padding
    input_masks[j] += padding

tokens_tensor = torch.tensor(tokens)
segments_tensors = torch.tensor(segments)
input_masks_tensors = torch.tensor(input_masks)

text_hashCodes = textNet(tokens_tensor, segments_tensors, input_masks_tensors)  
a = text_hashCodes[0].detach().numpy()
