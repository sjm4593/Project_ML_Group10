from typing import List, Optional, Union
from torch.nn import CrossEntropyLoss, MSELoss

import torch
from transformers import PreTrainedModel, PretrainedConfig
import gc


class AdaboostConfig(PretrainedConfig):
    model_type = "adaboost"

    def __init__(
        self,
        models: List[PretrainedConfig],
        model_votes: List[float],
        *args,
        **kwargs,
    ) -> PretrainedConfig:
        self.models = models
        self.model_votes = model_votes


class AdaboostModel(PreTrainedModel):
    config_class = AdaboostConfig

    def __init__(
        self, config: AdaboostConfig, models: List[PreTrainedModel], *inputs, **kwargs
    ):
        super().__init__(config, *inputs, **kwargs)
        self.models = models

    def forward(self, inputs, labels):
        outputs = []
        for model, vote in zip(self.models, self.config.model_votes):
            output: torch.Tensor = model(inputs, labels=labels)
            (loss, logits) = output

            outputs.append(logits * vote)
        logits = torch.sum(torch.stack(outputs), dim=0)

        loss_fct = CrossEntropyLoss()  # -100 index = padding token
        masked_lm_loss = loss_fct(
            logits.view(-1, self.models[0].config.vocab_size), labels.view(-1)
        )

        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

        return masked_lm_loss, logits
