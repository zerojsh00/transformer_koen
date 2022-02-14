import torch
import torch.nn as nn

import numpy as np

import argparse
from dataloader import BUILD_VOCAB_FROM_TRAIN
from seq2seq_transformer import Seq2SeqTransformer
from trainer import Trainer
from timeit import default_timer as timer


def argument_parsing():
    parser = argparse.ArgumentParser(description="Transformer Argparser",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 데이터 관련 설정
    parser.add_argument("-dtdir", "--data_dir", default="./raw_data",
            help="Data Dir")

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

    # 학습 설정
    parser.add_argument("-bt","--BATCH_SIZE", type=int, default=16, # 1차 테스트에서는 16으로 설정함
                   help="Mini batch size")
    parser.add_argument("-epochs","--NUM_EPOCHS", type=int, default=30, 
                help="Total training step")

    # 기타 설정
    parser.add_argument("-srclang", "--SRC_LANGUAGE", default="ko",
            help="Source language")
    parser.add_argument("-tgtlang", "--TGT_LANGUAGE", default="en",
            help="Target language")
    parser.add_argument("-svdir", "--save_dir", default="./saved_model",
            help="Path to save model")
    parser.add_argument("-logdir", "--log_dir", default="./trainlog",
            help="Path to record trainlog")

    args = parser.parse_args()
    return args

def main(args) :

    # 데이터 적재와 모델 저장, 그리고 로그 기록을 위한 폴더 생성
    from pathlib import Path
    if not Path(args.data_dir).exists():
        Path(args.data_dir).mkdir()
    
    if not Path(args.save_dir).exists():
        Path(args.save_dir).mkdir()

    if not Path(args.log_dir).exists():
        Path(args.log_dir).mkdir()

    NUM_EPOCHS = args.NUM_EPOCHS
    SRC_LANGUAGE = args.SRC_LANGUAGE
    TGT_LANGUAGE = args.TGT_LANGUAGE
    NUM_ENCODER_LAYERS = args.n_layer
    NUM_DECODER_LAYERS = args.n_layer
    BATCH_SIZE = args.BATCH_SIZE
    EMB_SIZE = args.d_model
    NHEAD = args.n_head
    FFN_HID_DIM = args.d_f

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    import sys
    print(sys.version)
    print(f"디바이스 : {DEVICE}")
    
    print("어휘 생성 중 ...")
    _, vocab_transform = BUILD_VOCAB_FROM_TRAIN()
    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])

    print("모델을 빌딩하는 중 ...")
    torch.manual_seed(0)

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=args.PAD_IDX)
    
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    trainer = Trainer(model=transformer, 
                        optimizer=optimizer, 
                        loss_fn=loss_fn, 
                        BATCH_SIZE=BATCH_SIZE, 
                        SRC_LANGUAGE=SRC_LANGUAGE, 
                        TGT_LANGUAGE=TGT_LANGUAGE, 
                        DEVICE=DEVICE)
    
    train_losses = []
    valid_losses = []
    
    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
        train_loss = trainer.train_epoch()
        end_time = timer()

        valid_loss = trainer.evaluate()

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Valid loss: {valid_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

        # 에폭마다 loss의 history를 남김
        np.savetxt("log_dir/loss_history.txt", np.array([train_losses, valid_losses]), fmt="%.4e")

        if epoch == 1 :
            LEAST_VALID_LOSS = valid_loss
            torch.save({'model' : transformer.state_dict(), 'optimizer':optimizer.state_dict()} , './BEST_MODEL.tar')

        else :
            if LEAST_VALID_LOSS > valid_loss :
                print("모델 갱신 : valid loss {}".format(valid_loss))
                LEAST_VALID_LOSS = valid_loss
            torch.save({'model' : transformer.state_dict(), 'optimizer':optimizer.state_dict()} , './BEST_MODEL.tar')

if __name__ == "__main__":
    args = argument_parsing()
    main(args)