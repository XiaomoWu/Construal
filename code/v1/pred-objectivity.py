import datatable as dt
import pickle
import pytorch_lightning as pl
import spacy
import torch
import torchmetrics

from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pyarrow.feather import read_feather
from pytorch_lightning import LightningModule, LightningDataModule
from pyarrow.feather import write_feather

class ObjDataset(Dataset):
    def __init__(self, text_type: str):
        '''
        text_type: "project_desc" or "title"
        '''
        super().__init__()

        pjson = read_feather('/home/yu/OneDrive/Construal/data/pjson.feather')
        self.pids = pjson['pid'].tolist()

        if text_type == 'project_desc':
            self.text = pjson['project_desc'].tolist()
        elif text_type == 'title':
            self.text = pjson['title'].tolist()

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        return self.pids[idx], self.text[idx]


def collate_fn(batch):
    pids, texts = zip(*batch)

    '''
    # by chunk
    input_tokens = tokenizer(list(texts), max_length=64, padding=True, truncation=True,
                             return_tensors='pt', return_overflowing_tokens=True, return_length=True)
    '''

    # by sentences
    sents = []
    doc_lens = []

    for doc in nlp.pipe(texts):
        sent = list(doc.sents)
        sent = [s.text for s in sent]
        doc_lens.append(len(sent))
        sents.extend(sent)

    input_tokens = tokenizer(sents, max_length=64, padding=True, truncation=True, return_tensors='pt', return_overflowing_tokens=False)

    return pids, input_tokens, doc_lens


class ObjModel(LightningModule):
    def __init__(self):
        super().__init__()

        self.model = AutoModelForSequenceClassification.from_pretrained(
            'distilbert-base-uncased')
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, input_ids, attention_mask, doc_lens):
        # if one doc have multiple chunks/sents, average its softmaxed logits
        y = self.model(input_ids, attention_mask).logits.argmax(dim=-1).float()
        y = torch.split(y, doc_lens)
        y = [doc.mean().int().tolist() for doc in y]

        return y

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


# -----------------
# predict
# -----------------

# init sentencizer
nlp = spacy.load('en_core_web_sm', exclude=['ner', 'tagger', 'lemmatizer', 'textcat'])


pl.seed_everything(42)

# init dataloader
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
dataset = ObjDataset(text_type='project_desc')
dataloader = DataLoader(dataset, batch_size=64, num_workers=4,
                        pin_memory=True, drop_last=False, collate_fn=collate_fn, shuffle=False)

# init model
model = ObjModel()
model.load_from_checkpoint('/home/yu/OneDrive/Construal/checkpoints/objectivity-model.ckpt')
model.freeze()

# start inferencing!
device = 'cuda:1'

model.to(device)
model.eval()

outputs = []
with torch.no_grad():
    for i, batch in enumerate(tqdm(dataloader)):

        pids, input_tokens, doc_lens = batch

        input_ids = input_tokens['input_ids'].to(device)
        attention_mask = input_tokens['attention_mask'].to(device)

        y = model(input_ids, attention_mask, doc_lens)
        outputs.append((pids, y))

# save results
pids = []
ys = []
for pid, y in outputs:
    pids.extend(pid)
    ys.extend(y)

objectivity_desc = dt.Frame(pid=pids, objectivity=ys)
write_feather(objectivity_desc.to_arrow(), 'data/objectivity/objectivity-predictions_desc.feather')