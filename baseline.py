import argparse
from ignite.metrics import Accuracy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizerFast, BertForSequenceClassification
from dataset import load_dataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--cuda", default=0, type=int)
parser.add_argument("--dataset", default="chnsenticorp", type=str)
parser.add_argument("--max_length", default=128, type=int)
parser.add_argument("--num_iter", default=10, type=int)
args = parser.parse_args()

pretrained = "hfl/chinese-macbert-base"
# pretrained = "hfl/chinese-macbert-large"
device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

config = BertConfig.from_pretrained(pretrained)
config.output_hidden_states = True
config.output_attentions = True

tokenizer = BertTokenizerFast.from_pretrained(pretrained, config=config)
model = BertForSequenceClassification.from_pretrained(pretrained, config=config).to(
    device
)
optimizer = optim.AdamW(
    [
        {"params": model.bert.parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 1e-2},
    ]
)

dataset = load_dataset()
dataset = dataset.map(
    lambda data: tokenizer(
        data["text"],
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_token_type_ids=False,
        return_attention_mask=False,
    ),
    batched=True,
    batch_size=4096,
)
dataset.set_format(type="torch", columns=["input_ids", "label"])
dataloader = DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, num_workers=12
)

accuracy = Accuracy()

for iter in range(args.num_iter):
    accuracy.reset()

    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        loss, logits, hidden_states, attentions = model(
            input_ids, labels=labels
        ).values()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy.update((logits, labels))

    model.save_pretrained(f"/data/pretrained/{pretrained}/iter_{iter:02d}")
    print(f"Iter: {iter}, Acc: {accuracy.compute()}")
