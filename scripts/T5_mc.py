from transformers import T5PreTrainedModel, T5Config, \
    T5_PRETRAINED_MODEL_ARCHIVE_MAP, T5Model, T5WithLMHeadModel
from torch.nn import CrossEntropyLoss, Linear
import torch

class T5ForMultipleChoice(T5PreTrainedModel):
    config_class = T5Config
    pretrained_model_archive_map = T5_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "t5"

    def __init__(self, config):
        super(T5ForMultipleChoice, self).__init__(config)

        self.t5 = T5WithLMHeadModel(config)
        #choose to use hidden states or softmaxed values for classification? currently softmaxed_values
        # self.classifier = Linear(config.d_model, config.vocab_size, bias=False)
 
        self.init_weights()

    def forward(self, **kwargs):
        
        num_choices = kwargs['encoder_input_ids'].shape[1]
        labels = kwargs.pop('labels')
        kwargs['encoder_input_ids'] = kwargs['encoder_input_ids'].view(-1, kwargs['encoder_input_ids'].size(-1))
        kwargs['encoder_attention_mask'] = kwargs['encoder_attention_mask'].view(-1, kwargs['encoder_attention_mask'].size(-1))
        kwargs['decoder_input_ids'] = kwargs['decoder_input_ids'].view(-1, kwargs['decoder_input_ids'].size(-1))
        kwargs['decoder_attention_mask'] = kwargs['decoder_attention_mask'].view(-1, kwargs['decoder_attention_mask'].size(-1))
        decoder_outputs,encoder_outputs = self.t5(**kwargs)
        print(decoder_outputs.shape)
        
        # logits = self.classifier(decoder_outputs)
        
        #currently using probability -softmaxed values of each time step for prediction
        shift_logits = decoder_outputs[...,:-1,:]
        softmaxed_probs = shift_logits.softmax()
        
        #TAKE INDICIES OF EACH TIME STEP HERE !!TODO
        
        #change back to (batch , num choices, seq len, dim)
        reshaped_logits = softmaxed_probs.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


