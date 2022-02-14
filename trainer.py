from tqdm import tqdm
from utils import create_mask
from torch.utils.data import DataLoader
from dataloader import GET_CUSTOM_KOEN_DATASET, collate_fn

class Trainer(object) :

    def __init__(self, model, optimizer, loss_fn, BATCH_SIZE, SRC_LANGUAGE, TGT_LANGUAGE, DEVICE) :
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
        self.BATCH_SIZE = BATCH_SIZE
        self.SRC_LANGUAGE = SRC_LANGUAGE
        self.TGT_LANGUAGE = TGT_LANGUAGE
        self.DEVICE = DEVICE

    def train_epoch(self):
        self.model.train()
        losses = 0
        train_iter = GET_CUSTOM_KOEN_DATASET(split='train', language_pair=(self.SRC_LANGUAGE, self.TGT_LANGUAGE))
        train_dataloader = DataLoader(train_iter, batch_size=self.BATCH_SIZE, collate_fn=collate_fn)

        for src, tgt in tqdm(train_dataloader):
            src = src.to(self.DEVICE)
            tgt = tgt.to(self.DEVICE)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

            logits = self.model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

            self. optimizer.zero_grad()

            tgt_out = tgt[1:, :]
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()

            self.optimizer.step()
            losses += loss.item()

        return losses / len(train_dataloader)


    def evaluate(self):
        self.model.eval()
        losses = 0

        val_iter = GET_CUSTOM_KOEN_DATASET(split='valid', language_pair=(self.SRC_LANGUAGE, self.TGT_LANGUAGE))
        val_dataloader = DataLoader(val_iter, batch_size=self.BATCH_SIZE, collate_fn=collate_fn)

        for src, tgt in tqdm(val_dataloader):
            src = src.to(self.DEVICE)
            tgt = tgt.to(self.DEVICE)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

            logits = self.model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

            tgt_out = tgt[1:, :]
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()

        return losses / len(val_dataloader)