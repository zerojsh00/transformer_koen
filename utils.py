import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    """
    (e.g.) 타겟 시퀀스 길이가 5토큰이라고 친다면, 

    >>> mask
    tensor([[ True, False, False, False, False], : 첫 토큰은 자기 자신만 attend 할 수 있음
        [ True,  True, False, False, False],
        [ True,  True,  True, False, False],
        [ True,  True,  True,  True, False],
        [ True,  True,  True,  True,  True]]) : 마지막 토큰은 모든 토큰을 attend 할 수 있음
    """

    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    """
    (e.g.) 타겟 시퀀스 길이가 5토큰이라고 친다면, 

    >>> mask
    tensor([[0., -inf, -inf, -inf, -inf], : 첫 토큰은 자기 자신만 attend 할 수 있음
        [0., 0., -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf],
        [0., 0., 0., 0., -inf],
        [0., 0., 0., 0., 0.]]) : 마지막 토큰은 모든 토큰을 attend 할 수 있음
    """

    return mask


def create_mask(src, tgt):
    """
    mask되지 않고 attend 할 수 있는 토큰은 False 또는 0 값으로 이루어지도록 처리함
    """

    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)
    # src_mask는 (src시퀀스길이 X src시퀀스길이)의 False들로 이루어진 행렬임

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)

    """
    (e.g.) 
    src_padding_mask 및 tgt_padding_mask는 아래와 같이 batch로 묶인 데이터들에 대해서 PAD_IDX에 해당하는 부분은 False 처리함

    tensor([[False, False, False, False, False, False,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True],

        [False, False, False, False, False, False, False, False, False, False,
         False, False,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True],

        [False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False]]) 
    """

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask