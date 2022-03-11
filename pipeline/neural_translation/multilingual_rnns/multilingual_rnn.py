# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from typing import List
from dataclasses import dataclass, field
from torch.nn import Embedding

from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.models import (
    FairseqMultiModel,
    register_model,
    register_model_architecture,
)
from .rnn import RNNEncoder, RNNDecoder, RNNModelConfig, \
    DEFAULT_MAX_TARGET_POSITIONS, DEFAULT_MAX_SOURCE_POSITIONS
# Bug fix for https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/2
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
# end

@dataclass
class MultilingualRNNModelConfig(RNNModelConfig):
    encoder_embeddings_sharing: str = field(
        default="",
        metadata={"help": "share encoder embeddings across some source languages"}
    )
    decoder_embeddings_sharing: str = field(
        default="",
        metadata={"help": "share decoder embeddings across some source languages"}
    )
    encoders_sharing: str = field(
        default="",
        metadata={"help": "share encoders across some source languages (supersedes encoder_embeddings_sharing)"}
    )
    decoders_sharing: str = field(
        default="",
        metadata={"help": "share decoders across some source languages (supersedes decoder_embeddings_sharing)"}
    )

@register_model('multilingual_rnn', dataclass=MultilingualRNNModelConfig)
class MultilingualRNNModel(FairseqMultiModel):
    """Train RNN models for multiple language pairs simultaneously.
    """

    def __init__(self, encoders, decoders):
        super().__init__(encoders, decoders)

    @classmethod
    def add_args(cls, parser):
        """Add criterion-specific arguments to the parser."""
        dc = getattr(cls, "__dataclass", None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc())

    @classmethod
    def build_model(cls, cfg: MultilingualRNNModelConfig, task):
        """Build a new model instance."""
        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb
        
        from fairseq.tasks.multilingual_translation import MultilingualTranslationTask
        assert isinstance(task, MultilingualTranslationTask)

        if not hasattr(cfg, 'max_source_positions'):
            cfg.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if not hasattr(cfg, 'max_target_positions'):
            cfg.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_langs = list(set([lang_pair.split('-')[0] for lang_pair in task.model_lang_pairs]))
        tgt_langs = list(set([lang_pair.split('-')[1] for lang_pair in task.model_lang_pairs]))

        # INITIALISATION - overrides of params if needed
        for name, langs in zip(["encoder", "decoder"], [src_langs, tgt_langs]):
            # If has shared encoder
            if isinstance(getattr(cfg, f'{name}s_sharing'), str) and getattr(cfg, f'{name}s_sharing') != "": 
                setattr(cfg, f'{name}s_sharing', [lang.split("-") for lang in getattr(cfg, f'{name}s_sharing').split(",")])
                setattr(cfg, f'{name}_embeddings_sharing', getattr(cfg, f'{name}s_sharing'))
            else: # If embedding is shared
                setattr(cfg, f'{name}s_sharing', [[lang] for lang in langs])
                if isinstance(getattr(cfg, f'{name}_embeddings_sharing'), str) and getattr(cfg, f'{name}_embeddings_sharing') != "":
                    print([lang.split('-') for lang in getattr(cfg, f'{name}_embeddings_sharing').split(",")])
                    setattr(cfg, f'{name}_embeddings_sharing', [lang.split('-') for lang in getattr(cfg, f'{name}_embeddings_sharing').split(",")])
                else:
                    # If not sharing encoder or sharing encoder embeddings, default value 
                    setattr(cfg, f'{name}_embeddings_sharing', [[lang] for lang in langs])


        # TESTING - Basic tests on values
        #raise Exception(task.__dict__, cfg)

        if task.training:
            for langs, name in zip([src_langs, tgt_langs], ["encoders_sharing", "decoders_sharing"]):
                test_langs_flat = [lang for lang_group in getattr(cfg, name) for lang in lang_group]

                if len(test_langs_flat) != len(set(test_langs_flat)):
                    raise ValueError(f'Lang must appear only once in {name}')
                if sorted(test_langs_flat) != sorted(langs):
                    raise ValueError(f'{name} must contain all langs, with the format: lang_not_shared,lang_shared1-lang_shared2: {test_langs_flat} vs {langs}')

        # 1) Build shared embeddings in dict, lang to shared embeddings. 
        embed_tokens_enc, embed_tokens_dec = {}, {}
        if cfg.share_all_embeddings:
            if cfg.encoder_embed_dim != cfg.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if cfg.decoder_embed_path and (
                    cfg.decoder_embed_path != cfg.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            cur_embed_tokens = FairseqMultiModel.build_shared_embeddings(
                dicts=task.dicts,
                langs=task.langs,
                embed_dim=cfg.encoder_embed_dim,
                build_embedding=build_embedding,
                pretrained_embed_path=cfg.encoder_embed_path,
            )
            for lang in task.lang:
                embed_tokens_enc[lang] = cur_embed_tokens
                embed_tokens_dec[lang] = cur_embed_tokens
            cfg.share_decoder_input_output_embed = True
        else:
            for langs in cfg.encoder_embeddings_sharing:
                cur_embed_tokens = (
                    FairseqMultiModel.build_shared_embeddings(
                        dicts={lang: task.dicts[lang] for lang in langs},
                        langs=langs,
                        embed_dim=cfg.encoder_embed_dim,
                        build_embedding=build_embedding,
                        pretrained_embed_path=cfg.encoder_embed_path,
                    )
                )
                for lang in langs:
                    embed_tokens_enc[lang] = cur_embed_tokens

            for langs in cfg.decoder_embeddings_sharing:
                cur_embed_tokens = (
                    FairseqMultiModel.build_shared_embeddings(
                        dicts={lang: task.dicts[lang] for lang in langs},
                        langs=langs,
                        embed_dim=cfg.decoder_embed_dim,
                        build_embedding=build_embedding,
                        pretrained_embed_path=cfg.decoder_embed_path,
                    )
                )
                for lang in langs:
                    embed_tokens_dec[lang] = cur_embed_tokens


        # 2) TODO: Store encoders/decoders per language - create association between each language and its items
        lang2encoders, lang2decoders = {}, {}
        # 2.1) Create encoders
        for langs in cfg.encoders_sharing:
            current_encoder = RNNEncoder(
                    dictionary=task.dicts['-'.join(langs)],
                    embed_dim=cfg.encoder_embed_dim,
                    hidden_size=cfg.encoder_hidden_size,
                    num_layers=cfg.encoder_layers,
                    dropout_in=(cfg.encoder_dropout_in if cfg.encoder_dropout_in >= 0 else cfg.dropout),
                    dropout_out=(cfg.encoder_dropout_out if cfg.encoder_dropout_out >= 0 else cfg.dropout),
                    bidirectional=cfg.encoder_bidirectional,
                    pretrained_embed=embed_tokens_enc[langs[0]],
                    rnn_type=cfg.rnn_type,
                    max_source_positions=cfg.max_source_positions)

            for lang in langs:
                lang2encoders[lang] = current_encoder

        # 2.2) Create decoders
        for langs in cfg.decoders_sharing:
            current_decoder = RNNDecoder(
                dictionary=task.dicts['-'.join(langs)],
                embed_dim=cfg.decoder_embed_dim,
                hidden_size=cfg.decoder_hidden_size,
                out_embed_dim=cfg.decoder_out_embed_dim,
                num_layers=cfg.decoder_layers,
                attention_type=cfg.attention_type,
                dropout_in=(cfg.decoder_dropout_in if cfg.decoder_dropout_in >= 0 else cfg.dropout),
                dropout_out=(cfg.decoder_dropout_out if cfg.decoder_dropout_out >= 0 else cfg.dropout),
                rnn_type=cfg.rnn_type,
                encoder_output_units=cfg.encoder_hidden_size,
                pretrained_embed=embed_tokens_dec[langs[0]],
                share_input_output_embed=cfg.share_decoder_input_output_embed,
                adaptive_softmax_cutoff=(
                    utils.eval_str_list(cfg.adaptive_softmax_cutoff,
                                        type=int)
                    if cfg.criterion == "adaptive_loss"
                    else None
                ),
                max_target_positions=cfg.max_target_positions,
                residuals=False,
            )
            for lang in langs:
                lang2decoders[lang] = current_decoder

        # 3) Stores encoders and decoders per lang-pair
        encoders, decoders = OrderedDict(), OrderedDict()
        for lang_pair in task.model_lang_pairs:
            encoders[lang_pair] = lang2encoders[lang_pair.split("-")[0]]
            decoders[lang_pair] = lang2decoders[lang_pair.split("-")[1]]

        return MultilingualRNNModel(encoders, decoders)

    def load_state_dict(self, state_dict, strict=True, args="none"):
        state_dict_subset = state_dict.copy()
        for k, _ in state_dict.items():
            assert k.startswith("models.")
            lang_pair = k.split(".")[1]
            if lang_pair not in self.models:
                del state_dict_subset[k]
        super().load_state_dict(state_dict_subset, strict=strict, args=args)


@register_model_architecture('multilingual_rnn', 'multilingual_lstm')
def multilingual_lstm(cfg):
    cfg.encoder_bidirectional = False
    cfg.rnn_type = "lstm"


@register_model_architecture('multilingual_rnn', 'multilingual_gru')
def multilingual_gru(cfg):
    cfg.encoder_bidirectional = False
    cfg.rnn_type = "gru"


@register_model_architecture('multilingual_rnn', 'multilingual_bilstm')
def multilingual_bilstm(cfg):
    cfg.encoder_bidirectional = True
    cfg.rnn_type = "lstm"


@register_model_architecture('multilingual_rnn', 'multilingual_bigru')
def multilingual_bigru(cfg):
    cfg.encoder_bidirectional = True
    cfg.rnn_type = "gru"
