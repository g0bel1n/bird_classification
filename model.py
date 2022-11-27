import torch.nn as nn
from transformers import ConvNextModel, ConvNextPreTrainedModel, Swinv2PreTrainedModel, Swinv2Model


class MyConvNext(ConvNextPreTrainedModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.convnext = ConvNextModel(self.config)
        self.head = nn.Sequential(
            nn.Linear(self.config.hidden_sizes[-1], 512),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(512, self.config.num_labels),
        )
        self.post_init()

    def forward(self, pixel_values, labels=None):

        outputs = self.convnext(pixel_values)

        logits = self.head(outputs[1])

        return logits

class MySwinV2(Swinv2PreTrainedModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.convnext = Swinv2Model(self.config)
        self.head = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(512, self.config.num_labels),
        )
        self.post_init()

    def forward(self, pixel_values, labels=None):

        outputs = self.convnext(pixel_values)

        logits = self.head(outputs[1])

        return logits