
import torch
import torch.nn as nn
import torch.optim as optim
from model import Encoder, Decoder, Seq2Seq

PAD, SOS, EOS = "<pad>", "<sos>", "<eos>"

# Read data
with open("data/conversations.txt", "r") as f:
    lines = [line.strip().split("\t") for line in f if line.strip()]
    questions, answers = zip(*lines)

def tokenize(text):
    return text.lower().split()

# Build vocab
def build_vocab(sentences):
    vocab = {PAD: 0, SOS: 1, EOS: 2}
    for sentence in sentences:
        for token in tokenize(sentence):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

SRC_vocab = build_vocab(questions)
TRG_vocab = build_vocab(answers)

def encode(sentence, vocab):
    return [vocab[SOS]] + [vocab.get(tok, 0) for tok in tokenize(sentence)] + [vocab[EOS]]

def pad_sequences(sequences):
    max_len = max(len(seq) for seq in sequences)
    return [seq + [0]*(max_len - len(seq)) for seq in sequences]

src_encoded = pad_sequences([encode(q, SRC_vocab) for q in questions])
trg_encoded = pad_sequences([encode(a, TRG_vocab) for a in answers])

src_tensor = torch.LongTensor(src_encoded).transpose(0, 1)
trg_tensor = torch.LongTensor(trg_encoded).transpose(0, 1)

input_dim = len(SRC_vocab)
output_dim = len(TRG_vocab)

enc = Encoder(input_dim, 64, 128)
dec = Decoder(output_dim, 64, 128)
model = Seq2Seq(enc, dec)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=0)

N_EPOCHS = 300
for epoch in range(N_EPOCHS):
    model.train()
    optimizer.zero_grad()
    output = model(src_tensor, trg_tensor)
    output_dim = output.shape[-1]
    output = output[1:].reshape(-1, output_dim)
    trg = trg_tensor[1:].reshape(-1)
    loss = criterion(output, trg)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "chatbot_model.pth")
torch.save(SRC_vocab, "src_vocab.pth")
torch.save(TRG_vocab, "trg_vocab.pth")
