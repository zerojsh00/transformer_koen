import os
import torch
import pickle
from torchtext.data.datasets_utils import _read_text_iterator, _RawTextIterableDataset
from torch.nn.utils.rnn import pad_sequence

import sentencepiece as spm
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List

# 특수 기호(symbol)와 인덱스를 정의함
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# 토큰들이 어휘집(vocab)에 인덱스 순서대로 잘 삽입되어 있는지 확인함
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

SRC_LANGUAGE = 'ko'
TGT_LANGUAGE = 'en'

def GET_CUSTOM_KOEN_DATASET(split, language_pair=('ko', 'en')):
    
    assert (split in ["train", "valid", "test"]), "split은 반드시 train, valid, test 중 하나여야 함"
    
    src_path = './raw_data/{}.ko'.format(split)
    trg_path = './raw_data/{}.en'.format(split)
    src_data_iter = _read_text_iterator(src_path)
    trg_data_iter = _read_text_iterator(trg_path)
    
    # 커스텀 데이터셋(AI-hub)의 라인 수. 데이터셋 구축 시 사전에 설정함
    NUM_LINES = {
    'train':1041109,
    'valid':130139,
    'test':130139
    }
    
    return _RawTextIterableDataset("KOEN_DATASET", NUM_LINES[split], zip(src_data_iter, trg_data_iter))


def BUILD_VOCAB_FROM_TRAIN():

    # 데이터 구축 시 사전에 생성해둠
    ko_spm = spm.SentencePieceProcessor()
    ko_vocab_file = "korean.model"
    ko_spm.load(ko_vocab_file)

    # 데이터 구축 시 사전에 생성해둠
    en_spm = spm.SentencePieceProcessor()
    en_vocab_file = "english.model"
    en_spm.load(en_vocab_file)


    # Place-holders
    token_transform = {}
    vocab_transform = {}

    token_transform[SRC_LANGUAGE] = ko_spm.encode_as_pieces
    token_transform[TGT_LANGUAGE] = en_spm.encode_as_pieces

    if os.path.isfile("./vocab_transform.pkl") :
        with open("./vocab_transform.pkl", "rb") as f :
            vocab_transform = pickle.load(f)

    else :
        
        # 토큰 목록을 생성하기 위한 헬퍼(helper) 함수
        def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
            language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

            for data_sample in data_iter:
                yield token_transform[language](data_sample[language_index[language]])


        for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
            # 학습용 데이터 반복자(iterator)
            train_iter = GET_CUSTOM_KOEN_DATASET(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
            # torchtext의 Vocab(어휘집) 객체 생성
            vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                            min_freq=1,
                                                            specials=special_symbols,
                                                            special_first=True)

        # UNK_IDX를 기본 인덱스로 설정함. 이 인덱스는 토큰을 찾지 못하는 경우에 반환됨
        # 만약 기본 인덱스를 설정하지 않으면 어휘집(Vocabulary)에서 토큰을 찾지 못하는 경우, RuntimeError가 발생됨
        for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
            vocab_transform[ln].set_default_index(UNK_IDX)
        
        # vocab을 저장함
        with open('./vocab_transform.pkl', 'wb') as f :
            pickle.dump(vocab_transform, f, pickle.HIGHEST_PROTOCOL)

    return (token_transform, vocab_transform)


# 순차적인 작업들을 하나로 묶는 헬퍼 함수
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# BOS/EOS를 추가하고 입력 순서(sequence) 인덱스에 대한 텐서를 생성하는 함수
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


# 데이터를 텐서로 조합(collate)하는 함수
def collate_fn(batch):

    token_transform, vocab_transform = BUILD_VOCAB_FROM_TRAIN()

    # 출발어(src)와 도착어(tgt) 원시 문자열들을 텐서 인덱스로 변환하는 변형(transform)
    text_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(token_transform[ln], # 토큰화(Tokenization)
                                                vocab_transform[ln], # 수치화(Numericalization)
                                                tensor_transform) # BOS/EOS를 추가하고 텐서를 생성

    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch