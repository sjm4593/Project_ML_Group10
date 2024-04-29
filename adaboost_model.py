from typing import List, Optional, Union

import torch
from transformers import PreTrainedModel, PretrainedConfig


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

    def forward(
        self,
        inputs,
    ):
        outputs = []
        for model, vote in zip(self.models, self.config.model_votes):
            output: torch.Tensor = model(inputs)
            (loss, logits, hidden_states, attentions) = output
            outputs.append(logits * vote)
        logits = torch.sum(torch.stack(outputs), dim=0)
        
        return logits
