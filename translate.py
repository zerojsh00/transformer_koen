import torch
import torch.nn as nn

import numpy as np
import pickle
import sentencepiece as spm
import argparse
from dataloader import BOS_IDX, BUILD_VOCAB_FROM_TRAIN
from seq2seq_transformer import Seq2SeqTransformer
from trainer import Trainer
from timeit import default_timer as timer

from utils import *
from dataloader import get_text_transform


def argument_parsing():
    parser = argparse.ArgumentParser(description="Transformer Argparser",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-unkidx", "--UNK_IDX", type=int, default=0,
            help="<unk> token index")
    parser.add_argument("-padidx", "--PAD_IDX", type=int, default=1,
            help="<pad> token index")
    parser.add_argument("-bosidx", "--BOS_IDX", type=int, default=2,
            help="<bos> token index")
    parser.add_argument("-eosidx", "--EOS_IDX", type=int, default=3,
            help="<eos> token index")
    

    # 모델 구조 설정
    parser.add_argument("-nl", "--n_layer", type=int, default=6,
                   help="Number of layers in Encoder / Decoder")
    parser.add_argument("-nh", "--n_head", type=int, default=8,
                   help="Number of heads in Multi-head Attention sublayer")
    parser.add_argument("-dm", "--d_model", type=int, default=512,
                   help="Dimension of model")
    parser.add_argument("-dk", "--d_k", type=int, default=64,
                   help="Dimension of key")
    parser.add_argument("-dv", "--d_v", type=int, default=64,
                   help="Dimension of value")
    parser.add_argument("-df", "--d_f", type=int, default=2048, 
                   help="Dimension of FFN")
    parser.add_argument("-drop", "--drop_rate", type=float, default=0.1,
                   help="Drop Rate")


    # 기타 설정
    parser.add_argument("-srclang", "--SRC_LANGUAGE", default="ko",
            help="Source language")
    parser.add_argument("-tgtlang", "--TGT_LANGUAGE", default="en",
            help="Target language")
    parser.add_argument("-svdir", "--save_dir", default="./saved_model",
            help="Path to save model")


    args = parser.parse_args()
    return args



def main(args) :

    with open("./vocab_transform.pkl", "rb") as f :
        vocab_transform = pickle.load(f)

    
    SRC_LANGUAGE = args.SRC_LANGUAGE
    TGT_LANGUAGE = args.TGT_LANGUAGE
    NUM_ENCODER_LAYERS = args.n_layer
    NUM_DECODER_LAYERS = args.n_layer
    EMB_SIZE = args.d_model
    NHEAD = args.n_head
    FFN_HID_DIM = args.d_f
    BOS_IDX = args.BOS_IDX

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    text_transform = get_text_transform()

    # 탐욕(greedy) 알고리즘을 사용하여 출력 순서(sequence)를 생성하는 함수
    def greedy_decode(model, src, src_mask, max_len, start_symbol):
        src = src.to(DEVICE)
        src_mask = src_mask.to(DEVICE)

        memory = model.encode(src, src_mask) # input 시퀀스를 트랜스포머 encoder에 통과시킴
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
        """
        >>> ys
        tensor([[2]]) # BOS_IDX
        """

        for i in range(max_len-1):
            memory = memory.to(DEVICE) # encoder를 통과한 input 시퀀스
            tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(DEVICE)
            
            # 현재까지 생성된 decoder의 output과 encoder를 통과한 input 시퀀스, 그리고 타겟마스크를 decoder에 입력함 
            out = model.decode(ys, memory, tgt_mask) 
            out = out.transpose(0, 1)

            # 디코더 결과를 linear layer에 통과시켜 타겟 vocab size 만큼 차원을 맞추어 줌
            prob = model.generator(out[:, -1]) # decoder output의 마지막 토큰 한 글자만 활용함
            _, next_word = torch.max(prob, dim=1) # the result tuple of two output tensors (max, max_indices)
            next_word = next_word.item() # 가장 probable한 단어 index를 next_word로 입력함

            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == EOS_IDX:
                break
        return ys


    # 입력 문장을 도착어로 번역하는 함수
    def translate(model: torch.nn.Module, src_sentence: str):
        model.eval()
        src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = greedy_decode(
            model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()

        # 어휘집에서 lookup_tokens 함수로 인덱스를 단어로 치환하여 문자열을 생성한 후 리턴함 
        return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")



    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])

    print("SRC_VOCAB_SIZE : {}".format(SRC_VOCAB_SIZE))
    print("TGT_VOCAB_SIZE : {}".format(TGT_VOCAB_SIZE))

    print("모델을 가져오는 중 ...")
    torch.manual_seed(0)

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
    
    checkpoint = torch.load(args.save_dir+"/BEST_MODEL.tar", map_location=DEVICE)
    transformer.load_state_dict(checkpoint["model"])

    en_spm = spm.SentencePieceProcessor()
    en_vocab_file = "english.model"
    en_spm.load(en_vocab_file)

    print()
    while(True) :
        print("문장 : ", end='')
        sentence = input()
        if sentence == "exit":
            break
        sentence = translate(transformer, sentence)
        sentence = en_spm.DecodePieces(sentence.split())
        print("번역 : {}".format(sentence))
        
    

if __name__ == "__main__":
    args = argument_parsing()
    main(args)
