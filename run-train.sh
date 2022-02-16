CUDA_VISIBLE_DEVICES=4 python main.py \
    -dtdir "./raw_data" \
    -unkidx 0 \
    -padidx 1 \
    -bosidx 2 \
    -eosidx 3 \
    -nl 6 \
    -nh 8 \
    -dm 512 \
    -dk 64 \
    -dv 64 \
    -df 2048 \
    -drop 0.1 \
    -bt 128 \
    -epochs 10000 \
    -srclang "ko" \
    -tgtlang "en" \
    -svdir "./saved_model" \
    -logdir "./trainlog" \
    > trainlog/train-koen.log
