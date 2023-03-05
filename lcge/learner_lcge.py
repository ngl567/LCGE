# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
from typing import Dict
import logging
from numpy.core.fromnumeric import _size_dispatcher
import torch
from torch import optim

from datasets_lcge import TemporalDataset
from optimizers_lcge import TKBCOptimizer, IKBCOptimizer
from models_lcge import LCGE
from regularizers_rule import N3, Lambda3

import sys

from regularizers_rule import RuleSim
import json

parser = argparse.ArgumentParser(
    description="Logic and Commonsense-Guided Temporal KGE"
)
parser.add_argument(
    '--dataset', type=str,
    help="Dataset name"
)
models = [
    'LCGE'
]
parser.add_argument(
    '--model', choices=models,
    help="Model in {}".format(models)
)
parser.add_argument(
    '--max_epochs', default=50, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid_freq', default=5, type=int,
    help="Number of epochs between each valid."
)
parser.add_argument(
    '--rank', default=100, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Batch size."
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--emb_reg', default=0., type=float,
    help="Embedding regularizer strength"
)
parser.add_argument(
    '--time_reg', default=0., type=float,
    help="Timestamp regularizer strength"
)
parser.add_argument(
    '--no_time_emb', default=False, action="store_true",
    help="Use a specific embedding for non temporal relations"
)
parser.add_argument(
    '--rule_reg', default=0., type=float,
    help="Rule regularizer strength"
)
parser.add_argument(
    '--weight_static', default=0., type=float,
    help="Weight of static score"
)


args = parser.parse_args()

dataset = TemporalDataset(args.dataset)

with open("./src_data/rulelearning/" + args.dataset + "/rule1_p1.json", 'r') as load_rule1_p1:
    rule1_p1 = json.load(load_rule1_p1)
with open("./src_data/rulelearning/" + args.dataset + "/rule1_p2.json", 'r') as load_rule1_p2:
    rule1_p2 = json.load(load_rule1_p2)

f = open("./src_data/rulelearning/" + args.dataset + "/rule2_p1.txt", 'r')
rule2_p1 = {}
for line in f:
    head, body1, body2, confi = line.strip().split("\t")
    head, body1, body2, confi = int(head), int(body1), int(body2), float(confi)
    if head not in rule2_p1:
        rule2_p1[head] = {}
    rule2_p1[head][(body1, body2)] = confi
f.close()

f = open("./src_data/rulelearning/" + args.dataset + "/rule2_p2.txt", 'r')
rule2_p2 = {}
for line in f:
    head, body1, body2, confi = line.strip().split("\t")
    head, body1, body2, confi = int(head), int(body1), int(body2), float(confi)
    if head not in rule2_p2:
        rule2_p2[head] = {}
    rule2_p2[head][(body1, body2)] = confi
f.close()

f = open("./src_data/rulelearning/" + args.dataset + "/rule2_p3.txt", 'r')
rule2_p3 = {}
for line in f:
    head, body1, body2, confi = line.strip().split("\t")
    head, body1, body2, confi = int(head), int(body1), int(body2), float(confi)
    if head not in rule2_p3:
        rule2_p3[head] = {}
    rule2_p3[head][(body1, body2)] = confi
f.close()

f = open("./src_data/rulelearning/" + args.dataset + "/rule2_p4.txt", 'r')
rule2_p4 = {}
for line in f:
    head, body1, body2, confi = line.strip().split("\t")
    head, body1, body2, confi = int(head), int(body1), int(body2), float(confi)
    if head not in rule2_p4:
        rule2_p4[head] = {}
    rule2_p4[head][(body1, body2)] = confi
f.close()

rules = (rule1_p1, rule1_p2, rule2_p1, rule2_p2, rule2_p3, rule2_p4)

sizes = dataset.get_shape()
print("sizes of dataset is:\t", sizes)
model = {
    'LCGE': LCGE(sizes, args.rank, rules, args.weight_static, no_time_emb=args.no_time_emb),
}[args.model]
model = model.cuda()


opt = optim.Adagrad(model.parameters(), lr=args.learning_rate)

emb_reg = N3(args.emb_reg)
time_reg = Lambda3(args.time_reg)
rule_reg = RuleSim(args.rule_reg)       # relation embedding reglu via rules

best_mrr = 0.
best_hit = 0.
early_stopping = 0

for epoch in range(args.max_epochs):
    examples = torch.from_numpy(
        dataset.get_train().astype('int64')
    )
    #print("\nexamples:\n", examples.size())

    model.train()
    if dataset.has_intervals():
        optimizer = IKBCOptimizer(
            model, emb_reg, time_reg, opt, dataset,
            batch_size=args.batch_size
        )
        optimizer.epoch(examples)

    else:
        optimizer = TKBCOptimizer(
            model, emb_reg, time_reg, rule_reg, opt,
            batch_size=args.batch_size
        )
        optimizer.epoch(examples)


    def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
        """
        aggregate metrics for missing lhs and rhs
        :param mrrs: d
        :param hits:
        :return:
        """
        m = (mrrs['lhs'] + mrrs['rhs']) / 2.
        h = (hits['lhs'] + hits['rhs']) / 2.
        return {'MRR': m, 'hits@[1,3,10]': h}

    if epoch < 0 or (epoch + 1) % args.valid_freq == 0:
        if dataset.has_intervals():
            valid, test, train = [
                dataset.eval(model, split, -1 if split != 'train' else 50000)
                for split in ['valid', 'test', 'train']
            ]
            print("valid: ", valid)
            print("test: ", test)
            print("train: ", train)

        else:
            valid, test, train = [
                avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
                for split in ['valid', 'test', 'train']
            ]
            print("epoch: ", epoch+1)
            print("valid: ", valid['MRR'])
            print("test: ", test['MRR'])
            print("train: ", train['MRR'])

            print("test hits@n:\t", test['hits@[1,3,10]'])
            if test['MRR'] > best_mrr:
                best_mrr = test['MRR']
                best_hit = test['hits@[1,3,10]']
                early_stopping = 0
            else:
                early_stopping += 1
            if early_stopping > 10:
                print("early stopping!")
                break

print("The best test mrr is:\t", best_mrr)
print("The best test hits@1,3,10 are:\t", best_hit)
