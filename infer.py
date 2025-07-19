
import torch
from model import Encoder, Decoder, Seq2Seq

PAD, SOS, EOS = "<pad>", "<sos>", "<eos>"

def tokenize(text):
    return text.lower().split()

SRC_vocab = torch.load("src_vocab.pth")
TRG_vocab = torch.load("trg_vocab.pth")
inv_trg_vocab = {v: k for k, v in TRG_vocab.items()}

input_dim = len(SRC_vocab)
output_dim = len(TRG_vocab)

enc = Encoder(input_dim, 64, 128)
dec = Decoder(output_dim, 64, 128)
model = Seq2Seq(enc, dec)
model.load_state_dict(torch.load("chatbot_model.pth"))
model.eval()

def encode(sentence, vocab):
    return [vocab[SOS]] + [vocab.get(tok, 0) for tok in tokenize(sentence)] + [vocab[EOS]]

def reply(sentence):
    input_ids = encode(sentence, SRC_vocab)
    input_tensor = torch.LongTensor(input_ids).unsqueeze(1)

    with torch.no_grad():
        hidden, cell = model.encoder(input_tensor)
        input_token = torch.LongTensor([TRG_vocab[SOS]])
        output_tokens = []
        for _ in range(20):
            output, hidden, cell = model.decoder(input_token, hidden, cell)
            top1 = output.argmax(1)
            if top1.item() == TRG_vocab[EOS]:
                break
            output_tokens.append(top1.item())
            input_token = top1

    return ' '.join([inv_trg_vocab[idx] for idx in output_tokens])

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        break
    print("Bot:", reply(user_input))
