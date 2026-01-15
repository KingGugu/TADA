import numpy as np


def get_linear_item_similarity(user_seq, max_item, max_len, xi=0.2, lam=0.1):
    # Exclude items from the validation set and test set to avoid data leakage, and use the nearly 50 items from the training set.
    user_seq = [seq[:-2][-max_len:] for seq in user_seq]

    num_sessions = len(user_seq)
    num_items = max_item
    X = np.zeros((num_sessions, num_items), dtype=np.float32)
    for session_idx, seq in enumerate(user_seq):
        for item in seq:
            if 1 <= item <= max_item:
                X[session_idx, item - 1] = 1.0

    X_T_X = np.matmul(X.T, X)
    I = np.eye(num_items, dtype=np.float32)
    X_T_X_reg = X_T_X + lam * I
    P = np.linalg.inv(X_T_X_reg)
    gamma = np.zeros(num_items, dtype=np.float32)
    for j in range(num_items):
        p_jj = P[j, j]
        if 1 - lam * p_jj <= xi:
            gamma[j] = lam
        else:
            gamma[j] = (1 - xi) / p_jj

    diag_gamma = np.diag(gamma)
    B_S = np.eye(num_items, dtype=np.float32) - np.matmul(P, diag_gamma)

    np.fill_diagonal(B_S, np.clip(np.diag(B_S), a_max=xi, a_min=None))

    return B_S


def get_LIS_topk(user_seq, head_items, tail_items, max_item, max_len, k, sim_threshold):
    print('Start Constructing the LIS Similarity Matrix')
    similarity_matrix = get_linear_item_similarity(user_seq, max_item, max_len)
    print('The LIS has been completed. Start filtering out items with low similarity and obtaining the top-k item set.')
    tail_topk_dict = {}
    head_topk_dict = {}

    for tail_item in tail_items:
        tail_idx = tail_item - 1
        sim_scores = similarity_matrix[tail_idx].copy()
        sim_scores[tail_idx] = -np.inf  # Exclude the item itself

        valid_indices = np.where(sim_scores >= sim_threshold)[0]
        if len(valid_indices) == 0:
            valid_indices = np.argsort(sim_scores)[::-1][:k]
        else:
            valid_scores = sim_scores[valid_indices]
            sorted_valid_idx = valid_indices[np.argsort(valid_scores)[::-1]]
            valid_indices = sorted_valid_idx[:k]

        topk_similar_set = {idx + 1 for idx in valid_indices}
        tail_topk_dict[tail_item] = topk_similar_set

    for head_item in head_items:
        head_idx = head_item - 1
        sim_scores = similarity_matrix[head_idx].copy()
        sim_scores[head_idx] = -np.inf

        valid_indices = np.where(sim_scores >= sim_threshold)[0]
        if len(valid_indices) == 0:
            valid_indices = np.argsort(sim_scores)[::-1][:k]
        else:
            valid_scores = sim_scores[valid_indices]
            sorted_valid_idx = valid_indices[np.argsort(valid_scores)[::-1]]
            valid_indices = sorted_valid_idx[:k]

        topk_similar_set = {idx + 1 for idx in valid_indices}
        head_topk_dict[head_item] = topk_similar_set

    return head_topk_dict, tail_topk_dict
