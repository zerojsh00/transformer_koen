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
    # torchtext에서 translation dataset(Multi30k 등)을 load 하는 방식을 참조함
    
    assert (split in ["train", "valid", "test"]), "split은 반드시 train, valid, test 중 하나여야 함"
    
    src_path = './raw_data/{}.ko'.format(split)
    trg_path = './raw_data/{}.en'.format(split)

    src_data_iter = _read_text_iterator(src_path)
    trg_data_iter = _read_text_iterator(trg_path)
    
    # _read_text_iterator 함수는 "문자열\n"로 이루어진 파일을 읽어들여 "문자열"을 return하는 이터레이터를 생성하는 제너레이터임
    # def _read_text_iterator(path):
    #     with io.open(path, encoding="utf8") as f:
    #         for row in f:
    #             yield row


    # 커스텀 데이터셋(AI-hub)의 라인 수. 데이터셋 구축 시 사전에 설정함
    NUM_LINES = {
    'train':1041109,
    'valid':130139,
    'test':130139
    }
    
    return _RawTextIterableDataset("KOEN_DATASET", NUM_LINES[split], zip(src_data_iter, trg_data_iter)) # ("한글 문장", "영어 문장") 을 리턴해주는 방식


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
    token_transform = {} # 한/영 토크나이저를 담는 딕셔너리
    vocab_transform = {} 

    # 센텐스피스 토크나이저
    token_transform[SRC_LANGUAGE] = ko_spm.encode_as_pieces
    token_transform[TGT_LANGUAGE] = en_spm.encode_as_pieces

    if os.path.isfile("./vocab_transform.pkl") : # 미리 구축한 어휘집(Vocab) 객체가 있다면 이를 활용함
        with open("./vocab_transform.pkl", "rb") as f :
            vocab_transform = pickle.load(f)

    else : # 토크나이즈된 학습 데이터들로 어휘집(Vocab)을 생성함
        
        # 토큰 목록을 생성하기 위한 헬퍼(helper) 함수
        def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
            language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

            for data_sample in data_iter:
                # (e.g.) 한국어의 경우 다음과 같이 yield 함 : ko_spm.encode_as_pieces(("한국어 문장 예시", "english sentence sample")[0]) -> ['▁한국어', '▁문장', '▁예', '시']
                # (e.g.) 영어의 경우 다음과 같이 yield 함 : en_spm.encode_as_pieces(("한국어 문장 예시", "english sentence sample")[1]) -> ['▁eng', 'lish', '▁sentence', '▁sample']
                yield token_transform[language](data_sample[language_index[language]]) 


        for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
            # 학습용 데이터 반복자(iterator)
            train_iter = GET_CUSTOM_KOEN_DATASET(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
            
            # torchtext의 Vocab(어휘집) 객체 생성
            # (i.g.) vocab_transform['ko']에는 한국어 문장 토큰들을 활용하여 만들어진 torchtext의 Vocab 객체가 저장됨
            vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                            min_freq=1,
                                                            specials=special_symbols,
                                                            special_first=True)

        # UNK_IDX를 기본 인덱스로 설정함. 이 인덱스는 토큰을 찾지 못하는 경우에 반환됨
        # 만약 기본 인덱스를 설정하지 않으면 어휘집(Vocabulary)에서 토큰을 찾지 못하는 경우, RuntimeError가 발생됨
        for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
            vocab_transform[ln].set_default_index(UNK_IDX) # 어휘집에 없는 OOV 토큰의 경우, default index가 나올 수 있도록 세팅함
        
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

token_transform, vocab_transform = BUILD_VOCAB_FROM_TRAIN()

def get_text_transform() :
    # 출발어(src)와 도착어(tgt) 원시 문자열들을 텐서 인덱스로 변환하는 변형(transform)
    text_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(token_transform[ln], # 토큰화 (e.g.) "한국어 문장 예제" -> ['▁한국어', '▁문장', '▁예', '제'] 
                                                vocab_transform[ln], # 수치화 (e.g.) ['▁한국어', '▁문장', '▁예', '제'] -> [8015, 15835, 690, 193]
                                                tensor_transform) # BOS/EOS를 추가하고 텐서를 생성 (e.g.) [8015, 15835, 690, 193] -> tensor([    2,  8015, 15835,   690,   193,     3])
    return text_transform

# 데이터를 텐서로 조합(collate)하는 함수
def collate_fn(batch):

    text_transform = get_text_transform() 
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX) # Pad a list of variable length Tensors with padding_value
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)

    """
    다음은 collate_fn이 batch를 어떻게 만들어주는지를 보여줌

    예제 1) 
    >>> a =  text_transform['ko']("한국어 문장 예제")
    tensor([    2,  8015, 15835,   690,   193,     3])

    예제 2) 
    >>> b = text_transform['ko']("배고픈데 얼른 퇴근한 후 식사를 해야겠다")
    tensor([    2, 18877, 13465,   735, 13575,  4848,    30,   129,  6397,   584,
            1577,     3])
    
    예제 3) 
    >>> c = text_transform['ko']("오미크론 변이 확산으로 코로나19 유행 규모가 급증하면서 16일 신규 확진자 수가 9만명을 넘었다.")
    tensor([    2,   264,   159,   728,  1358,   967,    12,  2205,    38,   714,
        27205,  4599,  6920,  2541,  7476,   205,   506,    11,  1248, 11889,
           66,  1002,   155,  9064, 17767,     4,     3])
    
    >>> src_batch.append(a)
    >>> src_batch.append(b)
    >>> src_batch.append(c)

    >>> pad_sequence(src_batch, padding_value=PAD_IDX)
    tensor([[    2,     2,     2],
            [ 8015, 18877,   264],
            [15835, 13465,   159],
            [  690,   735,   728],
            [  193, 13575,  1358],
            [    3,  4848,   967],
            [    1,    30,    12],
            [    1,   129,  2205],
            [    1,  6397,    38],
            [    1,   584,   714],
            [    1,  1577, 27205],
            [    1,     3,  4599],
            [    1,     1,  6920],
            [    1,     1,  2541],
            [    1,     1,  7476],
            [    1,     1,   205],
            [    1,     1,   506],
            [    1,     1,    11],
            [    1,     1,  1248],
            [    1,     1, 11889],
            [    1,     1,    66],
            [    1,     1,  1002],
            [    1,     1,   155],
            [    1,     1,  9064],
            [    1,     1, 17767],
            [    1,     1,     4],
            [    1,     1,     3]])
    """

    return src_batch, tgt_batch