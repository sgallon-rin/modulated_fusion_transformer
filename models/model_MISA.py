#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        : 2022/10/27 17:27
# @Author      : shenjl
# @Email       : shenjl@lr.pi.titech.ac.jp
# @File        : model_MISA.py
# @Description :
# Adapted from MISA(https://github.com/declare-lab/multimodal-deep-learning/tree/main/MISA)


import numpy as np
import random

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None


# let's define a simple model that can deal with multimodal variable length sequence
class Model_MISA(nn.Module):
    def __init__(self, args, vocab_size, pretrained_emb=None):
        super(Model_MISA, self).__init__()

        self.args = args
        self.text_size = args.word_embed_size  # args.embedding_size
        self.visual_size = 0  # do not use visual
        self.acoustic_size = args.audio_feat_size  # args.acoustic_size

        self.input_sizes = input_sizes = [self.text_size, self.visual_size, self.acoustic_size]
        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.visual_size), int(self.acoustic_size)]
        self.output_size = output_size = args.ans_size
        self.dropout_rate = dropout_rate = args.dropout
        self.activation = nn.ReLU()  # self.args.activation()
        self.tanh = nn.Tanh()

        rnn = nn.LSTM  # if self.args.rnncell == "lstm" else nn.GRU
        # defining modules - two layer bidirectional LSTM with layer norm in between

        self.pretrained_emb = pretrained_emb

        if self.args.use_bert:
            # Initializing a BERT bert-base-uncased style configuration
            bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
            self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)
        else:
            self.embed = nn.Embedding(len(vocab_size), input_sizes[0])
            self.trnn1 = rnn(input_sizes[0], hidden_sizes[0], bidirectional=True)
            self.trnn2 = rnn(2 * hidden_sizes[0], hidden_sizes[0], bidirectional=True)

        # self.vrnn1 = rnn(input_sizes[1], hidden_sizes[1], bidirectional=True)
        # self.vrnn2 = rnn(2 * hidden_sizes[1], hidden_sizes[1], bidirectional=True)

        self.arnn1 = rnn(input_sizes[2], hidden_sizes[2], bidirectional=True)
        self.arnn2 = rnn(2 * hidden_sizes[2], hidden_sizes[2], bidirectional=True)

        ##########################################
        # mapping modalities to same sized space
        ##########################################
        if self.args.use_bert:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t', nn.Linear(in_features=768, out_features=args.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(args.hidden_size))
        else:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t',
                                      nn.Linear(in_features=hidden_sizes[0] * 4, out_features=args.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(args.hidden_size))

        # self.project_v = nn.Sequential()
        # self.project_v.add_module('project_v',
        #                           nn.Linear(in_features=hidden_sizes[1] * 4, out_features=args.hidden_size))
        # self.project_v.add_module('project_v_activation', self.activation)
        # self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(args.hidden_size))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a',
                                  nn.Linear(in_features=hidden_sizes[2] * 4, out_features=args.hidden_size))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(args.hidden_size))

        ##########################################
        # private encoders
        ##########################################
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1',
                                  nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())

        # self.private_v = nn.Sequential()
        # self.private_v.add_module('private_v_1',
        #                           nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))
        # self.private_v.add_module('private_v_activation_1', nn.Sigmoid())

        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3',
                                  nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))
        self.private_a.add_module('private_a_activation_3', nn.Sigmoid())

        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())

        ##########################################
        # reconstruct
        ##########################################
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))
        # self.recon_v = nn.Sequential()
        # self.recon_v.add_module('recon_v_1', nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))

        ##########################################
        # shared space adversarial discriminator
        ##########################################
        if not self.args.use_cmd_sim:
            self.discriminator = nn.Sequential()
            self.discriminator.add_module('discriminator_layer_1',
                                          nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))
            self.discriminator.add_module('discriminator_layer_1_activation', self.activation)
            self.discriminator.add_module('discriminator_layer_1_dropout', nn.Dropout(dropout_rate))
            self.discriminator.add_module('discriminator_layer_2',
                                          nn.Linear(in_features=args.hidden_size, out_features=len(hidden_sizes)))

        ##########################################
        # shared-private collaborative discriminator
        ##########################################

        self.sp_discriminator = nn.Sequential()
        self.sp_discriminator.add_module('sp_discriminator_layer_1',
                                         nn.Linear(in_features=args.hidden_size, out_features=4))

        # Remember to change feature sizes for LA/LAV- sgallon
        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.args.hidden_size * 4,
                                                           out_features=self.args.hidden_size * 2))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_3',
                               nn.Linear(in_features=self.args.hidden_size * 2, out_features=output_size))

        self.tlayer_norm = nn.LayerNorm((hidden_sizes[0] * 2,))
        # self.vlayer_norm = nn.LayerNorm((hidden_sizes[1] * 2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[2] * 2,))

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.args.hidden_size, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        packed_sequence = pack_padded_sequence(sequence, lengths, batch_first=True)

        if self.args.rnncell == "lstm":
            packed_h1, (final_h1, _) = rnn1(packed_sequence)
        else:
            packed_h1, final_h1 = rnn1(packed_sequence)

        padded_h1, _ = pad_packed_sequence(packed_h1)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths)

        if self.args.rnncell == "lstm":
            _, (final_h2, _) = rnn2(packed_normed_h1)
        else:
            _, final_h2 = rnn2(packed_normed_h1)

        return final_h1, final_h2

    def alignment(self, sentences, visual, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask):

        batch_size = lengths.size(0)

        if self.args.use_bert:
            bert_output = self.bertmodel(input_ids=bert_sent,
                                         attention_mask=bert_sent_mask,
                                         token_type_ids=bert_sent_type)

            bert_output = bert_output[0]

            # masked mean
            masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
            mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)
            bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len

            utterance_text = bert_output
        else:
            # extract features from text modality
            sentences = self.embed(sentences)
            final_h1t, final_h2t = self.extract_features(sentences, lengths, self.trnn1, self.trnn2, self.tlayer_norm)
            utterance_text = torch.cat((final_h1t, final_h2t), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # extract features from visual modality
        # final_h1v, final_h2v = self.extract_features(visual, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
        # utterance_video = torch.cat((final_h1v, final_h2v), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # extract features from acoustic modality
        final_h1a, final_h2a = self.extract_features(acoustic, lengths, self.arnn1, self.arnn2, self.alayer_norm)
        utterance_audio = torch.cat((final_h1a, final_h2a), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # Shared-private encoders
        self.shared_private(utterance_text, utterance_audio)

        if not self.args.use_cmd_sim:
            # discriminator
            reversed_shared_code_t = ReverseLayerF.apply(self.utt_shared_t, self.args.reverse_grad_weight)
            # reversed_shared_code_v = ReverseLayerF.apply(self.utt_shared_v, self.args.reverse_grad_weight)
            reversed_shared_code_a = ReverseLayerF.apply(self.utt_shared_a, self.args.reverse_grad_weight)

            self.domain_label_t = self.discriminator(reversed_shared_code_t)
            # self.domain_label_v = self.discriminator(reversed_shared_code_v)
            self.domain_label_a = self.discriminator(reversed_shared_code_a)
        else:
            self.domain_label_t = None
            # self.domain_label_v = None
            self.domain_label_a = None

        self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)
        # self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)
        self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)
        self.shared_or_private_s = self.sp_discriminator(
            (self.utt_shared_t + self.utt_shared_a) / 2.0)  # if v: 3.0

        # For reconstruction
        self.reconstruct()

        # 1-LAYER TRANSFORMER FUSION  # need to modify if v
        h = torch.stack((self.utt_private_t, self.utt_private_a, self.utt_shared_t,
                        self.utt_shared_a), dim=0)
        h = self.transformer_encoder(h)
        h = torch.cat((h[0], h[1], h[2], h[3]), dim=1)
        o = self.fusion(h)
        return o

    def reconstruct(self, ):

        self.utt_t = (self.utt_private_t + self.utt_shared_t)
        # self.utt_v = (self.utt_private_v + self.utt_shared_v)
        self.utt_a = (self.utt_private_a + self.utt_shared_a)

        self.utt_t_recon = self.recon_t(self.utt_t)
        # self.utt_v_recon = self.recon_v(self.utt_v)
        self.utt_a_recon = self.recon_a(self.utt_a)

    def shared_private(self, utterance_t, utterance_a):

        # Projecting to same sized space
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        # self.utt_v_orig = utterance_v = self.project_v(utterance_v)
        self.utt_a_orig = utterance_a = self.project_a(utterance_a)

        # Private-shared components
        self.utt_private_t = self.private_t(utterance_t)
        # self.utt_private_v = self.private_v(utterance_v)
        self.utt_private_a = self.private_a(utterance_a)

        self.utt_shared_t = self.shared(utterance_t)
        # self.utt_shared_v = self.shared(utterance_v)
        self.utt_shared_a = self.shared(utterance_a)

    def forward(self, sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask):
        # batch_size = lengths.size(0)
        o = self.alignment(sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask)
        return o

