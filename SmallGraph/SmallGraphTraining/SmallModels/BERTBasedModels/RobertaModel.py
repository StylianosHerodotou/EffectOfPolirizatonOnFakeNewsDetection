from transformers.modeling_outputs  import SequenceClassifierOutput

from DatasetRepresentation.DataPreprocessing.BERTProcessing import roberta_column_name
from SmallGraph.SmallGraphTraining.SmallModels.SmallGraphModel import SmallGraphModel
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch
from transformers import  RobertaForSequenceClassification


class RobertaModel(torch.nn.Module):

    def __init__(self, is_part_of_ensemble=False):
        super(RobertaModel, self).__init__()
        sequence_model = RobertaForSequenceClassification.from_pretrained("roberta-base")
        self.config = sequence_model.config
        self.roberta = sequence_model.roberta
        self.classifier = sequence_model.classifier
        self.is_part_of_ensemble = is_part_of_ensemble

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        if (self.is_part_of_ensemble):
            return sequence_output

        logits = self.classifier(sequence_output)
        return logits


class Roberta(SmallGraphModel):
    def __init__(self, model_hyperparameters, is_part_of_ensemble=False):
        super().__init__()
        self.model = RobertaModel(is_part_of_ensemble=is_part_of_ensemble)

    def forward(self, data):
        data= data.extra_inputs[roberta_column_name]
        return self.model(**data)

    def find_loss(self, logits, data):
        labels = data.y
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return loss