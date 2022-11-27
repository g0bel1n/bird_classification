import os

import tqdm
from PIL import Image
from torchvision.transforms import (CenterCrop, Compose, GaussianBlur,
                                    Normalize, RandomAutocontrast,
                                    RandomRotation, RandomVerticalFlip, Resize,
                                    ToTensor)


def AugmentDataset(folder_path: str = "bird_dataset/train"):

    aug = Compose(
        [
            RandomAutocontrast(),
            GaussianBlur(7),
            RandomVerticalFlip(),
            RandomRotation(90),
        ]
    )

    for bird_type in os.listdir(folder_path):
        for image in tqdm(os.listdir(arbo := "/".join((folder_path, bird_type)))):
            img = Image.open(img_path := "/".join((arbo, image)))
            aug(img).save(f"{img_path[:-4]}_aug1.jpg"), aug(img).save(
                f"{img_path[:-4]}_aug2.jpg"
            )


class ConvNextDataTransforms:
    def __init__(self, test=False):
        self.base_transform = Compose(
            [
                Resize((384, 384)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.test = test

    def __call__(self, example_batch):
        if self.test:
            return self.base_transform(example_batch)

        example_batch["pixel_values"] = [
            self.base_transform(x.convert("RGB")) for x in example_batch["image"]
        ]
        example_batch["labels"] = example_batch["label"]
        return example_batch
