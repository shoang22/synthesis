from torch.utils.data import Dataset
import logging
import torch
# import sys
# sys.path.append("./")
from generation.utils import suppress_stderr, tokenizer_from_vocab
from rdkit import Chem

class RetroDataset(Dataset):
    def __init__(self, opt, transforms):
        super().__init__()
        self.logger = logging.getLogger("base")
        with open(opt["data_root"], "r") as f:
            self.data = f.readlines()[1:] # skip header

        self.transforms = transforms
    
    def __getitem__(self, idx):
        left, right = self.data[idx].split(",")
        src = left.strip()
        tgt = right.strip()

        if self.transforms:
            for t in self.transforms:
                src, tgt = t(src, tgt)
        
        assert tgt is not None, f"target at idx {idx} returned none after transform"
        return src, tgt

    def __len__(self):
        return len(self.data)
    
    def debug(self):
        for i in range(5):
            s, t = self[i]
            print(f"source: {s}")
            print(f"target: {t}")

class RetroTrainDataset(RetroDataset):
    def __init__(self, opt):
        super().__init__(opt, None)

class RetroValDataset(RetroDataset):
    def __init__(self, opt):
        transforms = [canonize]
        super().__init__(opt, transforms)

def canonize(product, reagents):
    answer = []
    reags = set(reagents.split("."))
    sms = set()
    with suppress_stderr():
        for r in reags:
            m = Chem.MolFromSmiles(r)
            if m is not None:
                sms.add(Chem.MolToSmiles(m))
        if len(sms):
            answer = sorted(list(sms))
    if len(answer) == 0:
        return

    answer = ".".join(answer) 
    return product, answer


class RetroCollator:
    def __init__(self, opt):
        self.src_tokenizer = tokenizer_from_vocab(opt["src_vocab"])
        self.tgt_tokenizer = tokenizer_from_vocab(opt["tgt_vocab"])

    def __call__(self, batch):
        batch_size = len(batch)
        src_vocab_size = len(self.src_tokenizer)
        tgt_vocab_size = len(self.tgt_tokenizer)

        left = []
        right = []
        for src, tgt in batch:
            left.append(src)
            right.append(tgt)
        
        nl = len(left[0])
        nr = len(right[0])
        for i in range(1, batch_size, 1):
            nl_a = len(left[i])
            nr_a = len(right[i])
            if nl_a > nl:
                nl = nl_a
            if nr_a > nr:
                nr = nr_a
        
        # add start symbol
        nr += 1

        x = torch.zeros((batch_size, nl)).type(torch.LongTensor)
        mx = torch.zeros((batch_size, nl)).type(torch.LongTensor)
        
        y = torch.zeros((batch_size, nr)).type(torch.LongTensor)
        my = torch.zeros((batch_size, nr)).type(torch.LongTensor)

        z = torch.zeros((batch_size, nr, tgt_vocab_size)).type(torch.LongTensor)

        for cnt in range(batch_size):
            product = left[cnt]
            reactants = "^" + right[cnt] + "$"
            for i, p in enumerate(product):
                try:
                    x[cnt, i] = self.src_tokenizer[p]
                except:
                    x[cnt, i] = 1

            mx[cnt, :i+1] = 1
            for i in range((len(reactants) - 1)):
                try:
                    y[cnt, i] = self.tgt_tokenizer[reactants[i]]
                except:
                    y[cnt, i] = 1

                try:
                    # 1s in position of correct word, 0 otherwise
                    z[cnt, i, self.tgt_tokenizer[reactants[i + 1]]] = 1
                except:
                    # 1 in <unk> token position 
                    z[cnt, i, 1] = 1

            my[cnt, :i+1] = 1

        assert ((x == 0).sum(dim=-1) == (mx == 0).sum(dim=-1)).all()

        return {
            "src": x,
            "tgt": y,
            "src_padding_mask": mx,
            "tgt_padding_mask": my,
            "gt": z 
        }


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    opt_d = {"data_path": "uspto-50k/patents40k_x5MSShuf.csv"}
    opt_c = {"src_vocab": "vocab/src.vocab", "tgt_vocab": "vocab/tgt.vocab"}

    train_ds = RetroTrainDataset(opt_d)
    val_ds = RetroValDataset(opt_d)

    train_ds.debug()
    print("train done")
    val_ds.debug()
    print("val done")

    collate_fn = RetroCollator(opt_c)

    train_dl = DataLoader(train_ds, batch_size=4, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=4, collate_fn=collate_fn)
    for dp in val_dl:
        print(dp)

    print("done")

