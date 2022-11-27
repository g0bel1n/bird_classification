import argparse
import os
from utils import get_train_val

import torch
from torchinfo import summary
from transformers import get_scheduler

from data import ConvNextDataTransforms
from model import MyConvNext, MySwinV2

# Training settings
parser = argparse.ArgumentParser(description="RecVis A3 training script")
parser.add_argument(
    "--data",
    type=str,
    default="bird_dataset",
    metavar="D",
    help="folder where data is located. train_images/ and val_images/ need to be found in the folder",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=16,
    metavar="B",
    help="input batch size for training (default: 64)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--experiment",
    type=str,
    default="experiment",
    metavar="E",
    help="folder where experiment outputs are located.",
)
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)


n_labels = 20
max_lr = 2e-4
model_name = "facebook/convnext-base-384-22k-1k"


train_loader, val_loader, labels = get_train_val(transform=ConvNextDataTransforms())

labels = labels[:-1]

model = MyConvNext.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
    ignore_mismatched_sizes=True,
)


print(summary(model))


optimizer = torch.optim.AdamW(model.parameters(), max_lr, weight_decay=0)


num_training_steps = args.epochs * len(train_loader)
scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)


criterion = torch.nn.CrossEntropyLoss(label_smoothing=.2)
criterion.cuda()

if use_cuda:
    print("Using GPU")
    model.cuda()
else:
    print("Using CPU")


def train(epoch):
    model.train()
    for batch_idx, batch in enumerate(train_loader):

        batch = {k: v.cuda() for k, v in batch.items()}

        optimizer.zero_grad()
        output = model(**batch)
        loss = criterion(output.view(-1, n_labels), batch["labels"].view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        scheduler.step()
        # print(len(train_loader))
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t lr: {}".format(
                    epoch,
                    batch_idx * len(batch["labels"]),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data.item(),
                    optimizer.param_groups[0]["lr"],
                )
            )


def validation():
    model.eval()
    validation_loss = 0
    correct = 0

    for batch_idx, batch in enumerate(val_loader):

        batch = {k: v.cuda() for k, v in batch.items()}

        output = model(**batch)
        validation_loss = criterion(
            output.view(-1, n_labels), batch["labels"].view(-1)
        )
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(batch["labels"].data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            validation_loss,
            correct,
            len(val_loader.dataset),
            100.0 * correct / len(val_loader.dataset),
        )
    )
    return correct / len(val_loader.dataset)


for epoch in range(1, args.epochs + 1):
    train(epoch)
    val = validation()

    if val > 0.94 or epoch == args.epochs:
        model_file = args.experiment + "/model_" + str(epoch) + ".pth"
        torch.save(model.state_dict(), model_file)
        print(
            "Saved model to "
            + model_file
            + ". You can run `python evaluate.py --model "
            + model_file
            + "` to generate the Kaggle formatted csv file\n"
        )
