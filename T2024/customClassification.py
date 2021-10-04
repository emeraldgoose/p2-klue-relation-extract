import torch
from transformers import BertPreTrainedModel, BertModel
from transformers import AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput


class bertCustomMultilabelClassifier(BertPreTrainedModel):

    """ Custom Multi labels classificer based on Bert """

    def __init__(self, config, num_labels=30):
        super(bertCustomMultilabelClassifier, self).__init__(config)
        self.num_labels = num_labels
        self.in_feature = 768
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.GRU = torch.nn.GRU(input_size=self.in_feature, hidden_size=self.in_feature,
                                dropout=config.hidden_dropout_prob, bidirectional=True)
        self.classifier = torch.nn.Linear(self.in_feature, num_labels)

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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        pooled_output, _ = self.GRU(pooled_output)

        logits = self.classifier(pooled_output)
        loss = None

        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits, ) + outputs[2:]
            return ((loss, ) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def main():
    config = AutoConfig.from_pretrained('klue/bert-base')
    # print(config)
    model = bertCustomMultilabelClassifier(config)
    device = torch.device('cpu')


if __name__ == "__main__":
    main()
