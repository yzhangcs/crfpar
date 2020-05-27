# -*- coding: utf-8 -*-

import os
from parser.utils import Embedding
from parser.utils.alg import cky, crf
from parser.utils.common import bos, eos, pad, unk
from parser.utils.corpus import Corpus, Treebank
from parser.utils.field import (BertField, CharField, ChartField, Field,
                                RawField)
from parser.utils.fn import build, factorize
from parser.utils.metric import BracketMetric

import torch
import torch.nn as nn
from transformers import BertTokenizer


class CMD(object):

    def __call__(self, args):
        self.args = args
        if not os.path.exists(args.file):
            os.mkdir(args.file)
        if not os.path.exists(args.fields) or args.preprocess:
            print("Preprocess the data")
            self.TREE = RawField('trees')
            self.WORD = Field('words', pad=pad, unk=unk,
                              bos=bos, eos=eos, lower=True)
            if args.feat == 'char':
                self.FEAT = CharField('chars', pad=pad, unk=unk,
                                      bos=bos, eos=eos, fix_len=args.fix_len,
                                      tokenize=list)
            elif args.feat == 'bert':
                tokenizer = BertTokenizer.from_pretrained(args.bert_model)
                self.FEAT = BertField('bert',
                                      pad='[PAD]',
                                      bos='[CLS]',
                                      eos='[SEP]',
                                      tokenize=tokenizer.encode)
            else:
                self.FEAT = Field('tags', bos=bos, eos=eos)
            self.CHART = ChartField('charts')
            if args.feat in ('char', 'bert'):
                self.fields = Treebank(TREE=self.TREE,
                                       WORD=(self.WORD, self.FEAT),
                                       CHART=self.CHART)
            else:
                self.fields = Treebank(TREE=self.TREE,
                                       WORD=self.WORD, POS=self.FEAT,
                                       CHART=self.CHART)

            train = Corpus.load(args.ftrain, self.fields)
            if args.fembed:
                embed = Embedding.load(args.fembed, args.unk)
            else:
                embed = None
            self.WORD.build(train, args.min_freq, embed)
            self.FEAT.build(train)
            self.CHART.build(train)
            torch.save(self.fields, args.fields)
        else:
            self.fields = torch.load(args.fields)
            self.TREE = self.fields.TREE
            if args.feat in ('char', 'bert'):
                self.WORD, self.FEAT = self.fields.WORD
            else:
                self.WORD, self.FEAT = self.fields.WORD, self.fields.POS
            self.CHART = self.fields.CHART
        self.criterion = nn.CrossEntropyLoss()

        args.update({
            'n_words': self.WORD.vocab.n_init,
            'n_feats': len(self.FEAT.vocab),
            'n_labels': len(self.CHART.vocab),
            'pad_index': self.WORD.pad_index,
            'unk_index': self.WORD.unk_index,
            'bos_index': self.WORD.bos_index,
            'eos_index': self.WORD.eos_index
        })

        print(f"Override the default configs\n{args}")
        print(f"{self.TREE}\n{self.WORD}\n{self.FEAT}\n{self.CHART}")

    def train(self, loader):
        self.model.train()

        for trees, words, feats, (spans, labels) in loader:
            self.optimizer.zero_grad()

            batch_size, seq_len = words.shape
            lens = words.ne(self.args.pad_index).sum(1) - 1
            mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
            mask = mask & mask.new_ones(seq_len-1, seq_len-1).triu_(1)
            s_span, s_label = self.model(words, feats)
            loss, _ = self.get_loss(s_span, s_label, spans, labels, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.args.clip)
            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()

        total_loss = 0
        metric = BracketMetric()

        for trees, words, feats, (spans, labels) in loader:
            batch_size, seq_len = words.shape
            lens = words.ne(self.args.pad_index).sum(1) - 1
            mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
            mask = mask & mask.new_ones(seq_len-1, seq_len-1).triu_(1)
            s_span, s_label = self.model(words, feats)
            loss, s_span = self.get_loss(s_span, s_label, spans, labels, mask)
            preds = self.decode(s_span, s_label, mask)
            preds = [build(tree,
                           [(i, j, self.CHART.vocab.itos[label])
                            for i, j, label in pred])
                     for tree, pred in zip(trees, preds)]
            total_loss += loss.item()
            metric([factorize(tree, self.args.delete, self.args.equal)
                    for tree in preds],
                   [factorize(tree, self.args.delete, self.args.equal)
                    for tree in trees])
        total_loss /= len(loader)

        return total_loss, metric

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()

        all_trees = []
        for trees, words, feats in loader:
            batch_size, seq_len = words.shape
            lens = words.ne(self.args.pad_index).sum(1) - 1
            mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
            mask = mask & mask.new_ones(seq_len-1, seq_len-1).triu_(1)
            s_span, s_label = self.model(words, feats)
            if self.args.mbr:
                s_span = crf(s_span, mask, mbr=True)
            preds = self.decode(s_span, s_label, mask)
            preds = [build(tree,
                           [(i, j, self.CHART.vocab.itos[label])
                            for i, j, label in pred])
                     for tree, pred in zip(trees, preds)]
            all_trees.extend(preds)

        return all_trees

    def get_loss(self, s_span, s_label, spans, labels, mask):
        span_mask = spans & mask
        span_loss, span_probs = crf(s_span, mask, spans, self.args.mbr)
        label_loss = self.criterion(s_label[span_mask], labels[span_mask])
        loss = span_loss + label_loss

        return loss, span_probs

    def decode(self, s_span, s_label, mask):
        pred_spans = cky(s_span, mask)
        pred_labels = s_label.argmax(-1).tolist()
        preds = [[(i, j, labels[i][j]) for i, j in spans]
                 for spans, labels in zip(pred_spans, pred_labels)]

        return preds
