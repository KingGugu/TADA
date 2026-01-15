# -*- coding: utf-8 -*-

import os
import torch
import argparse
import numpy as np
import datetime
from models import SASRecModel,GRU4Rec,FMLPRecModel
from datasets import SASRecDataset
from trainers import SASRecTrainer
from utils import EarlyStopping, get_user_seqs, check_path, set_seed
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from ht_process import classify_head_and_tail, classify_user_preference
from collections import defaultdict
from ht_process import classify_head_and_tail, classify_user_preference, build_head_tail_relation
from LIS import get_LIS_topk


def show_args_info(args):
    print("--------------------Configure Info:------------")
    with open(args.log_file, 'a') as f:
        for arg in vars(args):
            value = getattr(args, arg)

            info = f"{arg:<30} : {str(value):<35}"
            print(info)
            f.write(info + '\n')


def main():
    total_start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser()
    # system args
    parser.add_argument('--data_dir', default='../data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--data_name', default='Beauty', type=str)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--model_idx', default=5, type=int, help="model idenfier 10, 20, 30...")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    parser.add_argument("--no_cuda", action="store_true")

    # model args
    parser.add_argument("--model_name", default='SASRec', type=str)
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.2, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.2, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)

    # train args
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs")
    parser.add_argument("--patience", type=int, default=10, help="early stop patience")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=2024, type=int)
    parser.add_argument("--star_test", default=100, type=int)
    parser.add_argument("--full_sort", type=bool, default=True)
    parser.add_argument("--no_filters", type=bool, default=True) # for_FMLP-Rec

    # augment args
    parser.add_argument("--beta", type=float, default=0.3, help="beta value distributions for beta")
    parser.add_argument("--head_ratio", type=float, default=0.2)
    parser.add_argument("--rate_a", type=float, default=0.31)
    parser.add_argument("--rate_b", type=float, default=0.51)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--th", type=float, default=0.3)
    parser.add_argument('--aug', default=1, type=int, help="the use of augmentation")

    args = parser.parse_args()
    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    args.data_file = args.data_dir + args.data_name + '.txt'
    user_seq, max_item, valid_rating_matrix, test_rating_matrix = get_user_seqs(args.data_file)
    args.item_size = max_item + 2

    # save model args
    args_str = f'{args.model_name}-{args.data_name}-{args.model_idx}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')

    show_args_info(args)
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    # Division of Head-to-Tail Users/Items
    args.head_users = set()
    args.tail_users = set()
    args.head_items = set()
    args.tail_items = set()
    args.average_len, args.item_interaction_counts = classify_head_and_tail(user_seq, args.head_ratio, args.head_items,
                                                                            args.tail_items, args.head_users,
                                                                            args.tail_users)
    # Division of User Preferences
    args.user_preference, args.user_tail_ratios = classify_user_preference(user_seq, args.head_items, args.tail_items)
    # Construction of an Enhanced Candidate Set
    if not args.do_eval and args.aug == 1:
        # build the global relation
        args.head_rel, args.tail_rel = build_head_tail_relation(user_seq, args.head_items, args.tail_items,
                                                                args.max_seq_length)
        args.head_topk, args.tail_topk = get_LIS_topk(user_seq, args.head_items, args.tail_items,
                                                      max_item, args.max_seq_length,
                                                      args.top_k, args.th)
        # Merge Two Co-occurrence Sets
        args.head_combined = defaultdict(set)
        for item in args.head_items:
            combined = args.head_rel.get(item, set()).union(args.head_topk.get(item, set()))
            args.head_combined[item] = combined

        args.tail_combined = defaultdict(set)
        for item in args.tail_items:
            combined = args.tail_rel.get(item, set()).union(args.tail_topk.get(item, set()))
            args.tail_combined[item] = combined

            # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)
    my_path = os.path.join('weight/', f'{args.model_name}-{args.data_name}-') + 'original.pt'

    # training data for node classification
    train_dataset = SASRecDataset(args, user_seq, data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = SASRecDataset(args, user_seq, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = SASRecDataset(args, user_seq, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    if args.model_name=='SASRec':
        model = SASRecModel(args=args)
    elif args.model_name=='GRU4Rec':
        model = GRU4Rec(args=args)
    elif args.model_name=='FMLPRec':
        args.no_filters=False
        model = FMLPRecModel(args=args)

    trainer = SASRecTrainer(model, train_dataloader, eval_dataloader,
                            test_dataloader, args)
    if args.aug == 1:
        trainer.load(my_path)

    # Used to record the training time from epoch 0 to star_test epoch.
    epoch_times = []

    if args.do_eval:
        trainer.args.train_matrix = test_rating_matrix
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        scores, result_info = trainer.test(0, args.full_sort)

    else:
        early_stopping = EarlyStopping(args.checkpoint_path, patience=args.patience, verbose=True)
        for epoch in range(args.epochs):
            # Record the start time of the current epoch.
            epoch_start = datetime.datetime.now()
            trainer.train(epoch)

            epoch_end = datetime.datetime.now()
            epoch_duration = (epoch_end - epoch_start).total_seconds()
            if epoch <= args.star_test:
                epoch_times.append(epoch_duration)

            if epoch > args.star_test:
                scores, _ = trainer.valid(epoch, args.full_sort)
                early_stopping(np.array(scores[-1:]), trainer.model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
        trainer.args.train_matrix = test_rating_matrix
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0, args.full_sort)

    # calculate the total time
    total_end_time = datetime.datetime.now()
    total_duration = (total_end_time - total_start_time).total_seconds()

    # Calculate the average time from epoch 0 to epoch star_test
    avg_epoch_time = 0.0
    if epoch_times:
        avg_epoch_time = sum(epoch_times) / len(epoch_times)

    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')
        f.write(f"Total running time: {total_duration:.2f} seconds\n")
        if epoch_times:
            f.write(f"Average training time for epochs 0 to {args.star_test}: {avg_epoch_time:.2f} seconds/epoch\n")
        else:
            f.write("No training epochs recorded (either in eval mode or star_test is negative)\n")

    print(f"Total running time: {total_duration:.2f} seconds")
    if epoch_times:
        print(f"Average training time for epochs 0 to {args.star_test}: {avg_epoch_time:.2f} seconds/epoch")


main()
