# -*- coding: utf-8 -*-

import torch
from utils import neg_sample, get_sample_negs
import numpy as np
import random
from torch.utils.data import Dataset


class SASRecDataset(Dataset):

    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, target_neg, answer):

        seq_set = set(items)

        pad_len = self.max_len - len(input_ids)

        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        cur_rec_tensors = (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(target_neg, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
        )

        return cur_rec_tensors

    def _data_augmentation(self, aug_id1, aug_id2, aug_pos, aug_neg):
        pad_len1 = self.max_len - len(aug_id1)
        aug_id1 = [0] * pad_len1 + aug_id1
        aug_id1 = aug_id1[-self.max_len:]
        assert len(aug_id1) == self.max_len

        pad_len2 = self.max_len - len(aug_id2)
        aug_id2 = [0] * pad_len2 + aug_id2
        aug_id2 = aug_id2[-self.max_len:]
        assert len(aug_id2) == self.max_len

        pad_len = self.max_len - len(aug_pos)
        aug_pos = [0] * pad_len + aug_pos
        aug_pos = aug_pos[-self.max_len:]
        assert len(aug_pos) == self.max_len

        pad_len = self.max_len - len(aug_neg)
        aug_neg = [0] * pad_len + aug_neg
        aug_neg = aug_neg[-self.max_len:]
        assert len(aug_neg) == self.max_len

        aug_tensors = (
            torch.tensor(aug_id1, dtype=torch.long),
            torch.tensor(aug_id2, dtype=torch.long),
            torch.tensor(aug_pos, dtype=torch.long),
            torch.tensor(aug_neg, dtype=torch.long)
        )
        return aug_tensors

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]
        seq_set = set(items)

        assert self.data_type in {"train", "valid", "test"}

        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]

            target_neg = []
            for _ in input_ids:
                target_neg.append(neg_sample(seq_set, self.args.item_size))

            answer = [0]  # no use
            rec_batch = self._data_sample_rec_task(user_id, items, input_ids, target_pos, target_neg, answer)
            if self.args.aug == 1:
                aug_ids_1, aug_ids_2, aug_pos, aug_neg = self.augment(input_ids, target_pos, target_neg)
                aug_batch = self._data_augmentation(aug_ids_1, aug_ids_2, aug_pos, aug_neg)
            else:
                aug_batch = ()  # No augmentation, only provide a placeholder
            return rec_batch, aug_batch

        elif self.data_type == "valid":
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]
            target_neg = []
            for _ in input_ids:
                target_neg.append(neg_sample(seq_set, self.args.item_size))
            return self._data_sample_rec_task(user_id, items, input_ids, target_pos, target_neg, answer)


        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]
            target_neg = []
            for _ in input_ids:
                target_neg.append(neg_sample(seq_set, self.args.item_size))
            return self._data_sample_rec_task(user_id, items, input_ids, target_pos, target_neg, answer)

    def __len__(self):
        return len(self.user_seq)

    def augment(self, seq, target_pos, target_neg):

        p_insert = 1 - len(seq) / self.max_len

        if random.random() < p_insert:
            aug_1, aug_2, aug_pos, aug_neg = self.insert(seq, target_pos, target_neg)
            return aug_1, aug_2, aug_pos, aug_neg
        else:
            aug_pos = target_pos
            aug_neg = target_neg
            aug_1, aug_2 = self.substitute(seq)
            return aug_1, aug_2, aug_pos, aug_neg

    def substitute(self, seq):
        aug_1 = seq[:]
        aug_2 = []
        p = random.uniform(self.args.rate_a, self.args.rate_b)

        for item in seq:
            if item in self.args.head_items:
                if self.args.head_combined[item] and random.random() < p:
                    aug_item = random.choice(list(self.args.head_combined[item]))
                    aug_2 += [aug_item]
                else:
                    aug_2 += [item]
            else:
                aug_2 += [item]

        return aug_1, aug_2

    def insert(self, seq, target_pos, target_neg):
        aug_1, aug_2, aug_pos, aug_neg = [], [], [], []
        p = random.uniform(self.args.rate_a, self.args.rate_b)

        for i, item in enumerate(seq):
            if item in self.args.tail_items and random.random() < p and self.args.tail_combined[item]:
                aug_item = random.choice(list(self.args.tail_combined[item]))
                aug_1 += [item, item]
                aug_pos += [target_pos[i], target_pos[i]]
                aug_neg += [target_neg[i], target_neg[i]]
                aug_2 += [aug_item, item]
            else:
                aug_1 += [item]
                aug_2 += [item]
                aug_pos += [target_pos[i]]
                aug_neg += [target_neg[i]]

        return aug_1, aug_2, aug_pos, aug_neg
