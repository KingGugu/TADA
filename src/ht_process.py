import math
from collections import Counter
from typing import Dict, List, Set, DefaultDict
from collections import defaultdict


def classify_head_and_tail(user_seq: List[List[int]],
                           head_ratio: float,
                           head_items: Set[int],
                           tail_items: Set[int],
                           head_users: Set[int],
                           tail_users: Set[int]) -> int:
    user_seq = [seq[:-2] for seq in user_seq]
    item_cnt = Counter([i for seq in user_seq for i in seq])
    # Sort items in descending order of their occurrence count
    sorted_items = sorted(item_cnt, key=item_cnt.get, reverse=True)
    split_item = int(len(sorted_items) * head_ratio)

    item_threshold = item_cnt[sorted_items[split_item - 1]] if split_item > 0 else 0
    min_item_interaction = item_cnt[sorted_items[-1]] if sorted_items else 0

    head_items.clear()
    head_items.update(sorted_items[:split_item])
    tail_items.clear()
    tail_items.update(sorted_items[split_item:])

    # ---------------- user ----------------
    user_cnt = {uid: len(seq) for uid, seq in enumerate(user_seq)}
    sorted_users = sorted(user_cnt, key=user_cnt.get, reverse=True)
    split_user = int(len(sorted_users) * head_ratio)

    user_threshold = user_cnt[sorted_users[split_user - 1]] if split_user > 0 else 0

    min_user_seq_length = user_cnt[sorted_users[-1]] if sorted_users else 0

    head_users.clear()
    head_users.update(sorted_users[:split_user])
    tail_users.clear()
    tail_users.update(sorted_users[split_user:])

    print(f"Item classification threshold: The minimum occurrence count of head items is {item_threshold}")
    print(f"User classification threshold: The minimum sequence length of head users is {user_threshold}")
    print(f"Minimum item interaction count: {min_item_interaction}")  # New: Minimum item interaction count
    print(f"Minimum user sequence length: {min_user_seq_length}")  # New: Minimum user sequence length
    print(f"Number of head items: {len(head_items)}, Number of tail items: {len(tail_items)}")
    print(f"Number of head users: {len(head_users)}, Number of tail users: {len(tail_users)}")

    avg_len = int(math.floor(sum(len(seq) for seq in user_seq) / len(user_seq)))
    return avg_len, item_cnt


def classify_user_preference(
        user_seq: List[List[int]],
        head_items: Set[int],
        tail_items: Set[int]
):
    user_preference = []
    user_tail_ratios = []
    user_seq = [seq[:-2] for seq in user_seq]  # Avoid data leakage

    for seq in user_seq:

        # calculate the number of tail_item in seq
        tail_count = 0
        total_count = len(seq)

        for item in seq:
            if item in tail_items:
                tail_count += 1

        tail_ratio = tail_count / total_count if total_count > 0 else 0.0
        is_tail_preference = tail_ratio >= 0.5

        user_preference.append(is_tail_preference)
        user_tail_ratios.append(tail_ratio)

    return user_preference, user_tail_ratios


# Co-occurrence Relationships.
def build_head_tail_relation(user_seq: List[List[int]], head_items: Set[int], tail_items: Set[int], max_len):
    head_relation: DefaultDict[int, Set[int]] = defaultdict(set)
    tail_relation: DefaultDict[int, Set[int]] = defaultdict(set)
    user_seq = [seq[:-2][-max_len:] for seq in user_seq]  # Avoid data leakage

    for seq in user_seq:
        for idx, item in enumerate(seq):
            if item in head_items:
                if idx > 0 and seq[idx - 1] in tail_items:
                    head_relation[item].add(seq[idx - 1])

                if idx + 1 < len(seq) and seq[idx + 1] in tail_items:
                    head_relation[item].add(seq[idx + 1])
            if item in tail_items:
                if idx > 0:
                    prev_item = seq[idx - 1]
                    tail_relation[item].add(prev_item)
    return head_relation, tail_relation
