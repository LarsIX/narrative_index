import torch
import torch.nn as nn
from transformers import BertModel
from transformers.modeling_outputs import SequenceClassifierOutput

class CustomFinBERT(nn.Module):
        """
    Custom binary classification head on top of a BERT backbone (e.g., FinBERT),
    with support for class weighting and dropout regularization.

    Parameters
    ----------
    backbone : transformers.PreTrainedModel
        A pre-trained BERT-based model (e.g., FinBERT) serving as the encoder.
    class_weights : torch.Tensor
        Class weights used to balance the loss function (typically for imbalanced datasets).
    config : transformers.PretrainedConfig
        Configuration object containing dropout probability, number of labels, etc.

    Attributes
    ----------
    backbone : transformers.PreTrainedModel
        The BERT-based language model used for encoding text.
    dropout : nn.Dropout
        Dropout layer for regularization.
    classifier : nn.Linear
        Linear classification head mapping hidden states to output logits.
    loss_fn : nn.CrossEntropyLoss
        Cross-entropy loss function with class weights.

    Forward Parameters
    ------------------
    input_ids : torch.LongTensor
        Tensor of input token IDs (batch_size, sequence_length).
    attention_mask : torch.LongTensor, optional
        Tensor indicating which tokens should be attended to.
    token_type_ids : torch.LongTensor, optional
        Segment token indices to indicate different portions of the input.
    labels : torch.LongTensor, optional
        Ground truth labels for the input batch.

    Returns
    -------
    transformers.modeling_outputs.SequenceClassifierOutput
        Output object containing:
        - loss: torch.FloatTensor (if labels are provided)
        - logits: torch.FloatTensor
        - hidden_states: optional
        - attentions: optional
    """
    def __init__(self, backbone, class_weights, config):
        super(CustomFinBERT, self).__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.pooler_output  # [batch_size, hidden_dim]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None
        )
