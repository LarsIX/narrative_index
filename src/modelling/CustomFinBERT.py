import torch
import torch.nn as nn
from transformers import BertModel
from transformers.modeling_outputs import SequenceClassifierOutput


class CustomFinBERT(nn.Module):
    """
    Custom binary classification head on top of a BERT backbone (e.g., FinBERT),
    with support for class weighting and dropout regularization.
    """
    def __init__(self, backbone, class_weights, config):
        super(CustomFinBERT, self).__init__()
        self.backbone = backbone
        p = getattr(config, "classifier_dropout", None)
        if p is None:
            p = getattr(config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(p)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # <<< changed: make buffer non-persistent so it doesn't appear in state_dict
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float(), persistent=False)
        else:
            self.class_weights = None

        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        pooled_output = getattr(outputs, "pooler_output", None)
        if pooled_output is None:
            pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if labels.dtype != torch.long:
                labels = labels.long()
            labels = labels.view(-1)
            # <<< safe: use weight if present
            weight = getattr(self, "class_weights", None)
            loss = nn.functional.cross_entropy(logits, labels, weight=weight)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None)
        )
