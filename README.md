# PyTorch Vanilla Transformer NMT

이 프로젝트는 AI-hub에 공개된 한/영 문장 데이터셋을 활용하여 만든 번역기로, **Attention Is All You Need**[[1]](#1)의 PyTorch 구현체를 정리한 코드입니다. 모델에 대한 자세한 사항은 원 논문과 PyTorch Tutorial을 참고하세요.

<br/>

---

### Data preparation

raw data는 train/valid/test 데이터셋 각각에 대해서 한 줄에 한 문장씩으로 이루어진 파일들을 준비합니다. 

한국어 문장으로 이루어진 데이터는 `.ko`확장자로, 영어 문장으로 이루어진 데이터는 `.en`확장자로 저장하며, `raw_data` 폴더에 다음과 같이 저장합니다.

```
raw_data
└--train.ko
└--train.en
└--valid.ko
└--valid.en
└--test.ko
└--test.en
```

<br/>

원하는 방식으로 데이터의 경로를 수정할 수 있지만, 이 경우 default setting과 다름으로 인한 에러가 발생하지 않도록 주의하세요.

<br/>

사전에 준비한 데이터셋으로부터 한국어와 영어 각각에 대해 센텐스피스 토크나이저를 준비합니다. 각 파일은 루트 디렉토리에 `korean.model` 및 `english.model` 파일로 준비해둡니다. 역시나 원하는 위치로 경로를 수정할 수 있지만, default setting과 다름으로 인한 에러가 발생하지 않도록 주의하세요.

---
### Arguments for training

**Arguments for training**

| argument        | type    | description                                                  | default            |
| --------------- | ------------------------------------------------------------ | ------------------ | --------------- |
| `-dtdir` | `str` | Data Dir | `"./raw_data"` |
| `-unkidx` | `int` | <unk> token index | `0` |
| `-padidx`      | `int`   | <pad> token index                                       | `1` |
| `-bosidx`     | `int`   | <bos> token index                                       | `2` |
| `-eosidx`     | `int`   | <eos> token index                                      | `3` |
| `-nl` 	| `int` | Number of layers in Encoder / Decoder | `6`          |
| `-nh`       | `int`  | Number of Multi-head Attention sublayer | `8`             |
| `-dm`      | `int` | Dimension of model                                   | `512`           |
| `-dk`          | `int`   | Dimension of key | `64`       |
| `-dv` | `int` | Dimension of value | `64`           |
| `-df` | `int` | Dimension of FFN | `2048`        |
| `-drop` | `float` | Drop Rate | `0.1` |
| `-bt`    | `int` | Mini batch size  | `128` |
| `-epochs` | `int` | Total training step | *직접 설정해주세요* |
| `-srclang` | `str` | Source Language | `"ko"` |
| `-tgtlang` | `str` | Target language | `"en"` |
| `-svdir` | `str` | Path to save model | `"./saved_model"` |
| `-logdir` | `str` | Path to record trainlog | `"./trainlog"` |

<br/>

---

### Arguments for translating

**Arguments for translating**

| argument           | type  | description                                               | default                     |
| ------------------ | ----- | --------------------------------------------------------- | --------------------------- |
| `-unkidx` | `int` | <unk> token index | `0` |
| `-padidx`      | `int`   | <pad> token index                                       | `1` |
| `-bosidx`     | `int`   | <bos> token index                                       | `2` |
| `-eosidx`     | `int`   | <eos> token index                                      | `3` |
| `-nl` 	| `int` | Number of layers in Encoder / Decoder | `6`          |
| `-nh`       | `int`  | Number of Multi-head Attention sublayer | `8`             |
| `-dm`      | `int` | Dimension of model                                   | `512`           |
| `-dk`          | `int`   | Dimension of key | `64`       |
| `-dv` | `int` | Dimension of value | `64`           |
| `-df` | `int` | Dimension of FFN | `2048`        |
| `-drop` | `float` | Drop Rate | `0.1` |
| `-srclang` | `str` | Source Language | `"ko"` |
| `-tgtlang` | `str` | Target language | `"en"` |
| `-svdir` | `str` | Path to save model | `"./saved_model"` |



<br/>

---

### How to run

1. 필요한 패키지들을 install 합니다.
    ```shell
    pip install -r requirements.txt
    ```
    <br/>
    
2. 학습을 하고자 하면, 아래의 명령어를 실행합니다.
    ```shell
    sh run-train.sh
    ```
    <br/>
   학습 중 가장 낮은 validation loss로 기록 모델이 아래의 된경로에 `BEST_MODEL.tar` 파일로 저장됩니다.
    
    ```
    saved_model
    └--BEST_MODEL.tar
    ```
    <br/>

3. 학습을 완료하면, 아래의 명령어를 통해 번역기를 테스트 할 수 있습니다.
    ```shell
    python translate.py
    ```
    <br/>



---

### References

<a id="1">[1]</a> *Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.* ([https://arxiv.org/pdf/1706.03762.pdf](https://arxiv.org/pdf/1706.03762.pdf))

[2] *https://tutorials.pytorch.kr/beginner/translation_transformer.html*
