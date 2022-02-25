from dataloader import SRC_LANGUAGE, TGT_LANGUAGE
import time
import torch
import pickle
import sentencepiece as spm

from translate import translate
from seq2seq_transformer import Seq2SeqTransformer
from flask import Flask, jsonify, request


NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 2048
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("./vocab_transform.pkl", "rb") as f :
    vocab_transform = pickle.load(f)


SRC_VOCAB_SIZE = len(vocab_transform['ko'])
TGT_VOCAB_SIZE = len(vocab_transform['en'])
print("SRC_VOCAB_SIZE : {}".format(SRC_VOCAB_SIZE))
print("TGT_VOCAB_SIZE : {}".format(TGT_VOCAB_SIZE))


transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                            NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

checkpoint = torch.load("./saved_model/BEST_MODEL.tar", map_location=DEVICE)
transformer.load_state_dict(checkpoint["model"])

en_spm = spm.SentencePieceProcessor()
en_vocab_file = "english.model"
en_spm.load(en_vocab_file)


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    received_data = request.get_json()
    sentence = received_data['sentence']

    start = time.time()
    sentence = translate(transformer, sentence)
    sentence = en_spm.DecodePieces(sentence.split())
    end = time.time()

    return jsonify(
        translated_sentence = sentence, 
        time = str(end-start)
        )

if __name__ == '__main__':

    PORT = 5000
    app.run(host="127.0.0.1", debug=True, port=PORT)
