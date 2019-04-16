import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix, dok_matrix, lil_matrix


def cal_user(data):
    unique_users = []
    for line in data.itertuples():
        unique_users.append(line[1] - 1)


def split_data(user_item_matrix, split_ratio=(3, 1, 1)):
    users, items = user_item_matrix.shape
    train = dok_matrix(user_item_matrix.shape)
    validation = dok_matrix(user_item_matrix.shape)
    test = dok_matrix(user_item_matrix.shape)
    user_item_matrix = lil_matrix(user_item_matrix)
    for user in np.arange(users):
        items = list(user_item_matrix.rows[user])
        if len(items) >= 5:
            np.random.shuffle(items)
            train_count = round(len(items) * split_ratio[0] / sum(split_ratio))
            valid_count = round(len(items) * split_ratio[1] / sum(split_ratio))

            for i in items[0: train_count]:
                train[user, i] = 1
            for i in items[train_count: train_count + valid_count]:
                validation[user, i] = 1
            for i in items[train_count + valid_count:]:
                test[user, i] = 1
    # 负采样
    users, items = user_item_matrix.shape
    neg_user_item_matrix = {}
    test_user_item_matrix = {}
    validation_user_item_matrix = {}
    all_items = set(list(np.arange(items)))
    for u in range(users):
        test_user_item_matrix[u] = test.getrow(u).nonzero()[1]
    for u in range(users):
        validation_user_item_matrix[u] = validation.getrow(u).nonzero()[1]
    for u in range(users):
        neg_user_item_matrix[u] = list(all_items - set(train.getrow(u).nonzero()[1]))
    unique_test_users = cal_user((pd.DataFrame(test.toarray())))
    unique_users_validation = cal_user((pd.DataFrame(validation.toarray())))
    return train.todok(), neg_user_item_matrix, test.todok(), validation.todok(), test_user_item_matrix, validation_user_item_matrix, set(
        unique_test_users), set(unique_users_validation)
