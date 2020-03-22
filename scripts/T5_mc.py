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
        self.classifier = Linear(int((config.max_seq_len/2)-1), 1)
 
        self.init_weights()

    def forward(self, **kwargs):
        
        
        bs, num_choices, seq_len = kwargs['encoder_input_ids'].shape
        labels = kwargs.pop('labels')
        kwargs['encoder_input_ids'] = kwargs['encoder_input_ids'].view(-1, kwargs['encoder_input_ids'].size(-1))
        kwargs['encoder_attention_mask'] = kwargs['encoder_attention_mask'].view(-1, kwargs['encoder_attention_mask'].size(-1))
        kwargs['decoder_input_ids'] = kwargs['decoder_input_ids'].view(-1, kwargs['decoder_input_ids'].size(-1))
        kwargs['decoder_attention_mask'] = kwargs['decoder_attention_mask'].view(-1, kwargs['decoder_attention_mask'].size(-1))
        decoder_outputs,encoder_outputs = self.t5(**kwargs)
        
        
        #currently using probability -softmaxed values of each time step for prediction
        
        shift_labels = kwargs['decoder_input_ids'][...,1:]
        shift_logits = decoder_outputs[...,:-1,:]
        softmaxed_probs = shift_logits.softmax(dim=-1)
        
        #take indices of each time step
        seq_probs = torch.zeros(bs*num_choices,seq_len-1).cuda()
        for i in range(bs*num_choices):
            seq_probs[i]= softmaxed_probs[i,torch.arange(seq_len-1),shift_labels[i]]
        
        seq_probs = self.classifier(seq_probs)
        # seq_probs = seq_probs.prod(dim=-1)
        reshaped_probs = seq_probs.view(bs, num_choices)
        
        outputs = (reshaped_probs, decoder_outputs, encoder_outputs)  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # print(reshaped_probs[0][0])
            # print(labels)
            loss = loss_fct(reshaped_probs, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


