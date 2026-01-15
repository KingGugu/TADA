# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.distributions as dist

from tqdm import tqdm
from torch.optim import Adam
from utils import recall_at_k, ndcg_k, get_metric


class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader,
                 args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model

        if self.cuda_condition:
            self.model.cuda()
        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.beta_distribution = dist.Beta(torch.tensor([args.beta]), torch.tensor([args.beta]))

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort=full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort=full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list, answers, user_list):
        item_answers = np.array([
            item
            for ans_list in answers
            for item in ans_list
        ])
        head_user_mask = np.array([u in self.args.head_users for u in user_list])
        tail_user_mask = ~head_user_mask
        head_item_mask = np.array([item in self.args.head_items for item in item_answers])
        tail_item_mask = ~head_item_mask

        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_10, NDCG_10 = get_metric(pred_list, 10)
        HIT_head_item, NDCG_head_item = get_metric(pred_list[head_item_mask], 10)
        HIT_tail_item, NDCG_tail_item = get_metric(pred_list[tail_item_mask], 10)
        HIT_head_user, NDCG_head_user = get_metric(pred_list[head_user_mask], 10)
        HIT_tail_user, NDCG_tail_user = get_metric(pred_list[tail_user_mask], 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@10": f"{HIT_10:.4f}",
            "NDCG@10": f"{NDCG_10:.4f}",
            "HIT_head_item": f"{HIT_head_item:.4f}",
            "NDCG_head_item": f"{NDCG_head_item:.4f}",
            "HIT_tail_item": f"{HIT_tail_item:.4f}",
            "NDCG_tail_item": f"{NDCG_tail_item:.4f}",
            "HIT_head_user": f"{HIT_head_user:.4f}",
            "NDCG_head_user": f"{NDCG_head_user:.4f}",
            "HIT_tail_user": f"{HIT_tail_user:.4f}",
            "NDCG_tail_user": f"{NDCG_tail_user:.4f}"
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        scores = [HIT_10, NDCG_10, HIT_head_item, NDCG_head_item, HIT_tail_item, NDCG_tail_item, HIT_head_user,
                  NDCG_head_user, HIT_tail_user, NDCG_tail_user]
        return scores, str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list, user_list):
        k_list = [5, 10, 20]
        recall, ndcg = [], []
        for k in k_list:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))

        scores = [
            recall[0], ndcg[0],  # 5
            recall[1], ndcg[1],  # 10
            recall[2], ndcg[2],  # 20
        ]

        # ---------- Head / Tail User Metrics ----------
        head_user_mask = np.array([u in self.args.head_users for u in user_list])
        tail_user_mask = ~head_user_mask

        def recallk_ndcgk(mask, k):
            if mask.sum() == 0:
                return 0.0, 0.0
            sub_ans = answers[mask]
            sub_rec = pred_list[mask]
            return recall_at_k(sub_ans, sub_rec, k), ndcg_k(sub_ans, sub_rec, k)

        # The metrics of head user
        recall_hu5, ndcg_hu5 = recallk_ndcgk(head_user_mask, 5)
        recall_hu10, ndcg_hu10 = recallk_ndcgk(head_user_mask, 10)
        recall_hu20, ndcg_hu20 = recallk_ndcgk(head_user_mask, 20)

        # The metrics of tail user
        recall_tu5, ndcg_tu5 = recallk_ndcgk(tail_user_mask, 5)
        recall_tu10, ndcg_tu10 = recallk_ndcgk(tail_user_mask, 10)
        recall_tu20, ndcg_tu20 = recallk_ndcgk(tail_user_mask, 20)

        # ---------- Head / Tail items Metrics ----------
        item_answers = np.array([
            item
            for ans_list in answers
            for item in ans_list
        ])
        head_item_mask = np.array([item in self.args.head_items for item in item_answers])
        tail_item_mask = ~head_item_mask

        # The metrics of head items
        recall_hi5, ndcg_hi5 = recallk_ndcgk(head_item_mask, 5)
        recall_hi10, ndcg_hi10 = recallk_ndcgk(head_item_mask, 10)
        recall_hi20, ndcg_hi20 = recallk_ndcgk(head_item_mask, 20)

        # The metrics of tail items
        recall_ti5, ndcg_ti5 = recallk_ndcgk(tail_item_mask, 5)
        recall_ti10, ndcg_ti10 = recallk_ndcgk(tail_item_mask, 10)
        recall_ti20, ndcg_ti20 = recallk_ndcgk(tail_item_mask, 20)

        post_fix_k5 = {
            "Epoch": epoch,
            "Overall_HIT@5": f"{recall[0]:.4f}", "Overall_NDCG@5": f"{ndcg[0]:.4f}",
            "Head_Item_HIT@5": f"{recall_hi5:.4f}", "Head_Item_NDCG@5": f"{ndcg_hi5:.4f}",
            "Tail_Item_HIT@5": f"{recall_ti5:.4f}", "Tail_Item_NDCG@5": f"{ndcg_ti5:.4f}",
            "Head_User_HIT@5": f"{recall_hu5:.4f}", "Head_User_NDCG@5": f"{ndcg_hu5:.4f}",
            "Tail_User_HIT@5": f"{recall_tu5:.4f}", "Tail_User_NDCG@5": f"{ndcg_tu5:.4f}",
        }

        post_fix_k10 = {
            "Epoch": epoch,
            "Overall_HIT@10": f"{recall[1]:.4f}", "Overall_NDCG@10": f"{ndcg[1]:.4f}",
            "Head_Item_HIT@10": f"{recall_hi10:.4f}", "Head_Item_NDCG@10": f"{ndcg_hi10:.4f}",
            "Tail_Item_HIT@10": f"{recall_ti10:.4f}", "Tail_Item_NDCG@10": f"{ndcg_ti10:.4f}",
            "Head_User_HIT@10": f"{recall_hu10:.4f}", "Head_User_NDCG@10": f"{ndcg_hu10:.4f}",
            "Tail_User_HIT@10": f"{recall_tu10:.4f}", "Tail_User_NDCG@10": f"{ndcg_tu10:.4f}",
        }

        post_fix_k20 = {
            "Epoch": epoch,
            "Overall_HIT@20": f"{recall[2]:.4f}", "Overall_NDCG@20": f"{ndcg[2]:.4f}",
            "Head_Item_HIT@20": f"{recall_hi20:.4f}", "Head_Item_NDCG@20": f"{ndcg_hi20:.4f}",
            "Tail_Item_HIT@20": f"{recall_ti20:.4f}", "Tail_Item_NDCG@20": f"{ndcg_ti20:.4f}",
            "Head_User_HIT@20": f"{recall_hu20:.4f}", "Head_User_NDCG@20": f"{ndcg_hu20:.4f}",
            "Tail_User_HIT@20": f"{recall_tu20:.4f}", "Tail_User_NDCG@20": f"{ndcg_tu20:.4f}",
        }

        post_fix = f"{str(post_fix_k5)}\n{str(post_fix_k10)}\n{str(post_fix_k20)}"
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(post_fix + '\n')

        return scores, post_fix

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class SASRecTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader,
                 args):
        super(SASRecTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            args
        )

    def iteration(self, epoch, dataloader, full_sort=True, train=True):

        str_code = "train" if train else "valid"

        if train:
            self.model.train()
            rec_avg_loss = 0.0
            rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            for i, (rec_batch, aug_batch) in rec_cf_data_iter:
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                user_ids, input_ids, target_pos, target_neg, _ = rec_batch

                # ---------- recommendation task ---------------#
                sequence_output = self.model.encoder(input_ids)
                rec_loss = self.cross_entropy(sequence_output, target_pos, target_neg)
                joint_loss = rec_loss

                if self.args.aug == 1:
                    aug_batch = tuple(t.to(self.device) for t in aug_batch)
                    aug_ids_1, aug_ids_2, aug_pos, aug_neg = aug_batch
                    aug_out_1 = self.model.encoder(aug_ids_1)
                    aug_out_2 = self.model.encoder(aug_ids_2)
                    self_mix_loss = self.self_mix_learning(aug_out_1, aug_out_2, aug_pos, aug_neg)
                    joint_loss += self_mix_loss

                    # ----------- Tail-aware Cross Augmentation ----------------#
                    head_data, tail_data = self.split_batch_by_preference(sequence_output, target_pos, target_neg,
                                                                          input_ids, user_ids,
                                                                          aug_out_2, aug_ids_2, aug_pos, aug_neg)
                    head_loss = self.whole_mix_learning(*head_data)
                    tail_loss = self.whole_mix_learning(*tail_data)
                    whole_mixup_loss = tail_loss + head_loss
                    joint_loss += whole_mixup_loss

                self.optim.zero_grad()
                joint_loss.backward()
                self.optim.step()
                rec_avg_loss += rec_loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_cf_data_iter)),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            rec_data_iter = tqdm(enumerate(dataloader),
                                 desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                 total=len(dataloader),
                                 bar_format="{l_bar}{r_bar}")
            self.model.eval()
            pred_list = None
            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output = self.model.encoder(input_ids)
                    recommend_output = recommend_output[:, -1, :]
                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()

                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0

                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                        user_list = user_ids.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                        user_list = np.append(user_list, user_ids.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list, user_list)

            else:
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, _, answers, sample_negs = batch
                    recommend_output = self.model.encoder(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                        answer_list = answers.cpu().data.numpy()
                        user_list = user_ids.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                        user_list = np.append(user_list, user_ids.cpu().data.numpy(), axis=0)

                return self.get_sample_scores(epoch, pred_list, answer_list, user_list)

    def self_mix_learning(self, aug_out_1, aug_out_2, pos_ids, neg_ids):
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)

        alpha = self.beta_distribution.sample(torch.tensor([aug_out_1.shape[0]])).unsqueeze(-1).to(self.device)

        mixed_seq_out = alpha * aug_out_1 + (1 - alpha) * aug_out_2

        mix_pos_logits = (mixed_seq_out * pos_emb).sum(dim=-1).view(-1)
        mix_neg_logits = (mixed_seq_out * neg_emb).sum(dim=-1).view(-1)

        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  # [batch*seq_len]
        mix_rec_loss = torch.sum(
            - torch.log(torch.sigmoid(mix_pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(mix_neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return mix_rec_loss

    def whole_mix_learning(self, seq_out, pos_ids, neg_ids, input_ids):
        batch_size = seq_out.shape[0]
        if batch_size == 0:
            return torch.tensor(0.0, device=self.device)

        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)

        alpha = self.beta_distribution.sample(torch.tensor([seq_out.shape[0]])).unsqueeze(-1).to(self.device)
        indices = torch.randperm(seq_out.shape[0])

        mixed_seq_out = alpha * seq_out + (1 - alpha) * seq_out[indices]
        mixed_pos_emb = alpha * pos_emb + (1 - alpha) * pos_emb[indices]
        mixed_neg_emb = alpha * neg_emb + (1 - alpha) * neg_emb[indices]
        mix_pos_logits = (mixed_seq_out * mixed_pos_emb).sum(dim=-1).view(-1)
        mix_neg_logits = (mixed_seq_out * mixed_neg_emb).sum(dim=-1).view(-1)

        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  # [batch*seq_len]
        mix_rec_loss = torch.sum(
            - torch.log(torch.sigmoid(mix_pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(mix_neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return mix_rec_loss

    def split_batch_by_preference(
            self, seq_out, pos_ids, neg_ids, input_ids, user_ids,
            aug_seq_out, aug_ids_2, aug_pos, aug_neg
    ):
        user_ids_np = user_ids.cpu().numpy()
        is_tail_preference = np.array([self.args.user_preference[uid] for uid in user_ids_np])

        head_mask = ~is_tail_preference
        tail_mask = is_tail_preference
        head_mask = torch.tensor(head_mask, dtype=torch.bool, device=self.device)
        tail_mask = torch.tensor(tail_mask, dtype=torch.bool, device=self.device)

        head_seq = seq_out[head_mask]
        head_pos = pos_ids[head_mask]
        head_neg = neg_ids[head_mask]
        head_input = input_ids[head_mask]

        tail_seq = seq_out[tail_mask]
        tail_pos = pos_ids[tail_mask]
        tail_neg = neg_ids[tail_mask]
        tail_input = input_ids[tail_mask]

        # The enhanced sequences obtained using T-insert and T-substitute.
        aug_head_seq = aug_seq_out[head_mask]
        aug_head_pos = aug_pos[head_mask]
        aug_head_neg = aug_neg[head_mask]
        aug_head_input = aug_ids_2[head_mask]

        aug_tail_seq = aug_seq_out[tail_mask]
        aug_tail_pos = aug_pos[tail_mask]
        aug_tail_neg = aug_neg[tail_mask]
        aug_tail_input = aug_ids_2[tail_mask]

        # Merge the original sequence with the enhanced sequence.
        head_data = (
            torch.cat([head_seq, aug_head_seq], dim=0),
            torch.cat([head_pos, aug_head_pos], dim=0),
            torch.cat([head_neg, aug_head_neg], dim=0),
            torch.cat([head_input, aug_head_input], dim=0)
        )

        tail_data = (
            torch.cat([tail_seq, aug_tail_seq], dim=0),
            torch.cat([tail_pos, aug_tail_pos], dim=0),
            torch.cat([tail_neg, aug_tail_neg], dim=0),
            torch.cat([tail_input, aug_tail_input], dim=0)
        )

        return head_data, tail_data
