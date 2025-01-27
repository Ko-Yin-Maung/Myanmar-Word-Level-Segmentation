import re
import torch
from torch import nn
from TorchCRF import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim, t2i):
        super(BiLSTM_CRF, self).__init__()
        self.tag2idx = t2i
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_dim * 2, tagset_size)
        self.crf = CRF(tagset_size)

    def forward(self, x, tags=None, lengths=None):
        embeds = self.embedding(x)
        packed_embeds = nn.utils.rnn.pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        packed_lstm_out, _ = self.lstm(packed_embeds)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_lstm_out, batch_first=True)
        emissions = self.linear(lstm_out)

        if tags is not None:
            crf_loss = -self.crf(emissions, tags, mask=(tags != self.tag2idx['P']).byte())
            loss = crf_loss.mean()
            return loss
        else:
            return self.crf.decode(emissions, mask=(x != 0).byte())

def load_model(model_path, vocab_size, tagset_size, embedding_dim, hidden_dim, tag2inx):
    model = BiLSTM_CRF(vocab_size, tagset_size, embedding_dim, hidden_dim, tag2inx)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_vocab_and_tags(vocab_path, tags_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = {word.strip(): i for i, word in enumerate(f.readlines())}
    
    with open(tags_path, 'r', encoding='utf-8') as f:
        tags = [tag.strip() for tag in f.readlines()]
        tag2idx = {tag: i for i, tag in enumerate(tags)}
        idx2tag = {i: tag for tag, i in tag2idx.items()}    
    return vocab, tag2idx, idx2tag

burmese_consonant = 'က-အ'
burmese_others = 'ဣဤဥဦဧဩဪဿ၌၍၏၊။~!@#$%^&*(_+/}{[.,:;/-`'
def syllable_segment(text):
    line = re.sub(r'[\s\u200b]+', ' ', text)
    line = re.sub(r'([a-zA-Z0-9၀-၉]+)', r' \1 ', line)
    line = re.sub("(?<![္])([" + burmese_consonant + "])(?![့်])|([" + burmese_others + "])", r" \1\2", line).strip()
    line = re.sub(r'\s+', ' ', line)
    return line.split()

def tags_predicion(model, text, vocab, idx2tag, tag2idx, unk_token='S'):
    syllables = syllable_segment(text)
    sentence = [vocab.get(word, vocab.get(word, tag2idx[unk_token])) for word in syllables]
    input_tensor =  torch.LongTensor([sentence])
    with torch.no_grad():
        ## Get emissions from the model
        embeds = model.embedding(input_tensor)
        lstm_out, _ = model.lstm(embeds)
        emissions = model.linear(lstm_out)
        mask = (input_tensor != 0).byte()
        
        ## Use CRF's viterbi_decode to get the predicted tags
        predicted_tags = model.crf.viterbi_decode(emissions, mask)

    predicted_tags = [idx2tag[idx] for idx in predicted_tags[0]]
    return syllables, predicted_tags


## Paths and Hyperparameters
model_path = 'model/word_segment.pth'
vocab_path = 'model/vocab'
tags_path = 'model/tags'
EMBEDDING_DIM = 128
HIDDEN_DIM = 256

## Load vocabulary, tags and Model
vocab, tag2idx, idx2tag = load_vocab_and_tags(vocab_path, tags_path)
model = load_model(model_path, len(vocab), len(tag2idx), EMBEDDING_DIM, HIDDEN_DIM, tag2idx)

def word_segment(input_text): 
    syllables, predicted_tags = tags_predicion(model, input_text, vocab, idx2tag, tag2idx)
    segmented_sentence = ""
    for word, tag in zip(syllables, predicted_tags):
        if tag == "E":
            segmented_sentence +=  word + " "
        elif tag == "B":
            segmented_sentence +=  " " + word
        elif tag == "S":
            segmented_sentence += " " + word + " "
        else:
            segmented_sentence += word
    segmented_sentence = re.sub(r'\s+', ' ', segmented_sentence.strip())

    # print("Input Text:", input_text)
    # print("Predicted Tags:", predicted_tags)
    return segmented_sentence