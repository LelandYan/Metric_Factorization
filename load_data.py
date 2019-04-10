import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix


def  form_csr_matrix(data, n_users, n_items):
    validation_row = []
    validation_col = []
    unique_users = []
    validation_rating = []

    for line in data.itertuples():
        validation_row.append(line[1] - 1)
        unique_users.append(line[1] - 1)
        validation_col.append(line[2] - 1)
        validation_rating.append(1)

    validation_matrix = csr_matrix((validation_rating, (validation_row, validation_col)), shape=(n_users, n_items))
    return validation_matrix, unique_users


def load_data(path="data/filmtrust.dat", header=['user_id', 'item_id', 'rating'], sep="\t"):
    df = pd.read_csv(path, sep=sep, names=header, engine='python')
    return df


def load_ranking_data(df, test_size=0.2):
    train_data, test_data = train_test_split(df, test_size=test_size)
    train_data, validation_data = train_test_split(train_data, test_size=0.25)

    n_users = df.user_id.max()
    n_items = df.item_id.max()
    # 稀疏矩阵的存储方式（Coordinate Format）COO
    train_row = []
    train_col = []
    train_rating = []

    train_dict = {}
    for line in train_data.itertuples():
        # 这里user_id，item_id都减1是为了将其与矩阵的索引相关联
        u = line[1] - 1
        i = line[2] - 1
        train_dict[(u, i)] = 1

    #count = 0
    # COO
    for u in range(n_users):
        for i in range(n_items):
            train_row.append(u)
            train_col.append(i)
            if (u, i) in train_dict.keys():
                #count = count + 1
                train_rating.append(1)
            else:
                train_rating.append(0)

    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))

    all_items = set(np.arange(n_items))

    # 负采样
    neg_user_item_matrix = {}
    #train_user_item_matrix = []

    for u in range(n_users):
        neg_user_item_matrix[u] = list(all_items - set(train_matrix.getrow(u).nonzero()[1]))
        #train_user_item_matrix.append(list(train_matrix.getrow(u).toarray()[0]))

    test_matrix, unique_users = form_csr_matrix(test_data, n_users, n_items)
    validation_matrix, unique_users_validation = form_csr_matrix(validation_data, n_users, n_items)

    test_user_item_matrix = {}

    for u in range(n_users):
        test_user_item_matrix[u] = test_matrix.getrow(u).nonzero()[1]

    validation_user_item_matrix = {}

    for u in range(n_users):
        validation_user_item_matrix[u] = validation_matrix.getrow(u).nonzero()[1]

    return train_matrix.todok(), neg_user_item_matrix, test_matrix.todok(), validation_matrix.todok(), test_user_item_matrix, validation_user_item_matrix, n_users, n_items, set(
        unique_users), set(unique_users_validation)
