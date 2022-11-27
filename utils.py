from datasets import load_dataset
import torch
from torch.utils.data.sampler import BatchSampler, RandomSampler


def collate_fn(batch):

    return {
        "pixel_values": torch.stack(batch[0]["pixel_values"]),
        "labels": torch.tensor(batch[0]["label"]),
    }


def get_train_val(transform, data_dir: str = "bird_dataset", batch_size=16):

    dataset = load_dataset("imagefolder", data_dir=data_dir, drop_labels=False)
    dataset.set_transform(transform)

    sampler_train = BatchSampler(
        RandomSampler(dataset["train"]), batch_size=batch_size, drop_last=False
    )

    sampler_val = BatchSampler(
        RandomSampler(dataset["validation"]),
        batch_size=batch_size // 2,
        drop_last=False,
    )
    return (
        torch.utils.data.DataLoader(
            dataset["train"],
            collate_fn=collate_fn,
            pin_memory=True,
            sampler=sampler_train,
        ),
        torch.utils.data.DataLoader(
            dataset["validation"],
            collate_fn=collate_fn,
            pin_memory=True,
            sampler=sampler_val,
        ),
        dataset["train"].features["label"].names,
    )


def get_test_dataset(transform, data_dir: str = "bird_dataset", batch_size=16):

    dataset = load_dataset("imagefolder", data_dir=data_dir, drop_labels=False)
    dataset.set_transform(transform)
    return torch.utils.data.DataLoader(
        dataset["test"], collate_fn=collate_fn, pin_memory=True, batch_size=batch_size
    )
