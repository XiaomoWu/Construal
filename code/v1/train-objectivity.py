import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader, Dataset, random_split
import torchmetrics
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pytorch_lightning import LightningModule, LightningDataModule


class ReviewDataset(Dataset):
    def __init__(self):
        super().__init__()

        with open('/home/yu/OneDrive/Construal/data/objectivity/plot.tok.gt9.5000', 'r') as f:
            obj = f.readlines()
            obj = [(1, s.strip()) for s in obj]

        with open('/home/yu/OneDrive/Construal/data/objectivity/quote.tok.gt9.5000', 'r') as f:
            subj = f.readlines()
            subj = [(0, s.strip()) for s in subj]

        self.data = obj + subj

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ReviewDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            'distilbert-base-uncased')

    def setup(self, stage=None):
        dataset = ReviewDataset()
        train_size = int(len(dataset) * 0.8)
        val_size = int(len(dataset)) - train_size

        # use partial dataset to train
        # self.train_dataset, self.val_dataset = random_split(
        #     dataset, [train_size, val_size])

        # use full dataset to train
        self.train_dataset = dataset
        self.val_dataset = random_split(dataset, [train_size, val_size])[1]

    def collate_fn(self, batch):
        labels, text = zip(*batch)
        labels = torch.tensor(labels, dtype=torch.long)
        input_tokens = self.tokenizer(
            list(text), max_length=512, padding=True, truncation=True, return_tensors='pt')

        return labels, input_tokens

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=64, num_workers=4, pin_memory=True, drop_last=False, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=64, num_workers=4, pin_memory=True, drop_last=False, collate_fn=self.collate_fn)


class ReviewModel(LightningModule):
    def __init__(self):
        super().__init__()

        self.model = AutoModelForSequenceClassification.from_pretrained(
            'distilbert-base-uncased')
        self.val_acc = torchmetrics.Accuracy()

    def training_step(self, batch, batch_idx):
        t, x = batch
        loss = self.model(**x, labels=t).loss
        return loss

    def validation_step(self, batch, batch_idx):
        t, x = batch
        y = torch.softmax(self.model(**x).logits, dim=-1)
        self.val_acc.update(y, t)

    def validation_epoch_end(self, outputs):
        acc = self.val_acc.compute()
        if self.global_rank == 0:
            print(f'Val accuracy: {acc*100}%')
        self.log('val/acc', acc)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        return optimizer


# init datamodule
datamodule = ReviewDataModule()

# init model
model = ReviewModel()

# init checkpoint callback
ckpt_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
    dirpath='checkpoints',
    monitor='val/acc',
    mode='max',
    save_last=False,
    save_top_k=1)

# init ddp plugin
ddp_plugin = pl.plugins.DDPPlugin(find_unused_parameters=False)

# init trainer
trainer = pl.Trainer(accelerator='ddp', gpus=[
                     0, 1], max_epochs=15, min_epochs=15, precision=16, callbacks=[ckpt_callback], plugins=ddp_plugin)

# start training
trainer.fit(model, datamodule=datamodule)
