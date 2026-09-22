"""TIGER model for CPU/CUDA/NPU porting experiments.

Lineage: Datawhale torch-rechub TIGERModel (T5ForConditionalGeneration subclass).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration


class TIGERModel(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_parallel = False
        self.device_map = None
        self.temperature = 1.0

    def set_hyper(self, temperature: float) -> None:
        self.temperature = float(temperature)

    def ranking_loss(self, lm_logits: torch.Tensor, labels: Optional[torch.Tensor]):
        if labels is None:
            return None
        t_logits = lm_logits / self.temperature
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        labels = labels.to(lm_logits.device)
        return loss_fct(t_logits.view(-1, t_logits.size(-1)), labels.view(-1))

    def _maybe_set_cuda_device(self, device_index) -> None:
        if self.model_parallel and torch.cuda.is_available():
            torch.cuda.set_device(device_index)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
        decoder_head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        if self.model_parallel:
            self._maybe_set_cuda_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)

        if self.model_parallel:
            self._maybe_set_cuda_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = decoder_outputs[0]

        if self.model_parallel:
            self._maybe_set_cuda_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)
        loss = self.ranking_loss(lm_logits, labels)

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


def build_small_t5_config(
    vocab_size: int,
    d_model: int = 128,
    d_ff: int = 256,
    num_layers: int = 2,
    num_heads: int = 4,
    d_kv: int = 32,
    dropout_rate: float = 0.1,
) -> T5Config:
    return T5Config(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        d_kv=d_kv,
        num_layers=num_layers,
        num_decoder_layers=num_layers,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="relu",
        is_encoder_decoder=True,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )


def build_paper_t5_config(vocab_size: int, dropout_rate: float = 0.1) -> T5Config:
    """TIGER paper seq2seq size targeting ~13M parameters.

    Paper: 4 enc/dec layers, 6 attention heads of dim 64 ⇒ d_model=384,
    MLP (d_ff)=1024, dropout=0.1. (With SID+user vocab this lands ~14–15M.)
    """
    return build_small_t5_config(
        vocab_size=vocab_size,
        d_model=384,
        d_ff=1024,
        num_layers=4,
        num_heads=6,
        d_kv=64,
        dropout_rate=dropout_rate,
    )


def build_xlt_t5_config(
    vocab_size: int = 1025,
    dropout_rate: float = 0.1,
    *,
    exact_xlt: bool = True,
    use_cache: bool = False,
) -> T5Config:
    """XiaoLongtaoo/TIGER defaults.

    exact_xlt=True: d_model=128, heads=6, d_kv=64 (inner_dim=384 ≠ d_model).
    On Ascend, fused attention may stack-smash with that layout — use
    exact_xlt=False (d_model=384) so d_model == heads * d_kv.
    """
    if exact_xlt:
        d_model, d_ff, d_kv, num_heads = 128, 1024, 64, 6
    else:
        # NPU-safe: match paper head geometry while keeping d_ff / depth
        d_model, d_ff, d_kv, num_heads = 384, 1024, 64, 6
    return T5Config(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        d_kv=d_kv,
        num_layers=4,
        num_decoder_layers=4,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="relu",
        is_encoder_decoder=True,
        use_cache=use_cache,
        pad_token_id=0,
        eos_token_id=0,
        decoder_start_token_id=0,
    )
