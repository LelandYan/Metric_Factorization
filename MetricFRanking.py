# _*_ coding: utf-8 _*_
import tensorflow as tf
import random
from sklearn.metrics import auc
from evaluation import *
from sklearn.metrics import precision_recall_curve

class MetricFRanking():

    def __init__(self, sess, num_users, num_items, learning_rate=0.1, epoch=200, N=500, batch_size=500):
        self.lr = learning_rate
        self.epochs = epoch
        self.N = N
        self.num_users = num_users
        self.num_items = num_items
        self.batch_size = batch_size
        self.clip_norm = 1
        self.sess = sess
        self.beta = 1.5  # 2.5#0.6#2.5#1.5

    def run(self, train_data, unique_users,unique_validation, neg_train_matrix, test_matrix,  validation_matrix,k=50):
        # train_data: 训练数据
        # unique_users: 用户ID列表
        # neg_train_matrix: 负面训练矩阵
        # test_matrix: 验证数据

        self.cf_user_input = tf.placeholder(dtype=tf.int32, shape=[None], name='cf_user_input')
        self.cf_item_input = tf.placeholder(dtype=tf.int32, shape=[None], name='cf_item_input')
        self.y = tf.placeholder(dtype=tf.float32, shape=[None], name='y')
        # m * N 的矩阵，初始化用户潜在因子矩阵
        U = tf.Variable(tf.random_normal([self.num_users, self.N], stddev=1 / (self.N ** 0.5)), dtype=tf.float32)
        # n * N 的矩阵，初始化物品潜在因子矩阵
        V = tf.Variable(tf.random_normal([self.num_items, self.N], stddev=1 / (self.N ** 0.5)), dtype=tf.float32)

        # 用户ID和对应的嵌入数据（潜在的用户向量）映射
        users = tf.nn.embedding_lookup(U, self.cf_user_input)
        # 物品ID和对应的嵌入数据（潜在的物品向量）映射
        pos_items = tf.nn.embedding_lookup(V, self.cf_item_input)

        # 获取对应的用户和物品距离差值
        self.pos_distances = tf.reduce_sum(tf.squared_difference(users, pos_items), 1, name="pos_distances")
        # dropout ，用于优化关于神经网络的过拟合的问题
        self.pred = tf.reduce_sum(tf.nn.dropout(tf.squared_difference(users, pos_items), 0.95), 1, name="pred")

        # 物品排序的损失函数
        self.loss = tf.reduce_sum((1 + 0.1 * self.y) * tf.square(
            (self.y * self.pred + (1 - self.y) * tf.nn.relu(self.beta * (1 - self.y) - self.pred))))

        self.optimizer = tf.train.AdagradOptimizer(self.lr).minimize(self.loss, var_list=[U, V])
        # 这里的clip_by_norm是指对梯度进行裁剪，通过控制梯度的最大范式，防止梯度爆炸的问题，是一种比较常用的梯度规约的方式。
        clip_U = tf.assign(U, tf.clip_by_norm(U, self.clip_norm, axes=[1]))
        clip_V = tf.assign(V, tf.clip_by_norm(V, self.clip_norm, axes=[1]))
        # initialize model
        init = tf.global_variables_initializer()

        # 将压缩稀疏矩阵转化为坐标矩阵，其中的坐标矩阵存储着(行，列，值)
        temp = train_data.tocoo()
        item = list(temp.col.reshape(-1))
        user = list(temp.row.reshape(-1))
        rating = list(temp.data)
        self.sess.run(init)
        # train and test the model
        sample_size = 0

        stop_num = 20
        t_stop_num = 0
        stop_threshold = 0.005
        pre_recall = 0
        k_Mat = [5, 10, 20, 30, 40, 50]
        r_recalls = np.zeros([1, 6])
        r_precisions = np.zeros([1, 6])
        epoch = 0
        while True:
            user_temp = user[:]
            item_temp = item[:]
            rating_temp = rating[:]
            epoch += 1
            # 从未评分的数据中采用负例
            user_append = []
            item_append = []
            values_append = []
            if epoch % 5 == 0:
                user_append = []
                item_append = []
                values_append = []
                for u in range(self.num_users):
                    if sample_size > len(neg_train_matrix[u]):
                        list_of_random_items = random.sample(neg_train_matrix[u], len(neg_train_matrix[u]))
                    else:
                        list_of_random_items = random.sample(neg_train_matrix[u], sample_size)
                    user_append += [u] * sample_size
                    item_append += list_of_random_items
                    values_append += [0] * sample_size
            #if user_append != None and item_append != None and values_append != None:
            item_temp += item_append
            user_temp += user_append
            rating_temp += values_append
            self.num_training = len(rating_temp)
            total_batch = int(self.num_training / self.batch_size)

            # 随机排序列表
            idxs = np.random.permutation(self.num_training)
            user_random = list(np.array(user_temp)[idxs])
            item_random = list(np.array(item_temp)[idxs])

            rating_random = list(np.array(rating_temp)[idxs])

            for i in range(total_batch):
                batch_user = user_random[i * self.batch_size:(i + 1) * self.batch_size]
                batch_item = item_random[i * self.batch_size:(i + 1) * self.batch_size]
                batch_rating = rating_random[i * self.batch_size:(i + 1) * self.batch_size]

                _, c, p, _, _ = self.sess.run((self.optimizer, self.loss, self.pos_distances, clip_U, clip_V),
                                              feed_dict={self.cf_user_input: batch_user,
                                                         self.cf_item_input: batch_item,
                                                         self.y: batch_rating})
            pred_ratings__valid = {}
            pred_ratings_valid = {}
            ranked_list_valid = {}
            p_at_ = []
            r_at_ = []

            for u in unique_validation:
                user_ids = []
                user_neg_items = neg_train_matrix[u]
                item_ids = []

                for j in user_neg_items:
                    item_ids.append(j)
                    user_ids.append(u)
                ratings = - self.sess.run([self.pos_distances]
                                          , feed_dict={self.cf_user_input: user_ids, self.cf_item_input: item_ids})[
                    0]
                neg_item_index = list(zip(item_ids, ratings))

                ranked_list_valid[u] = sorted(neg_item_index, key=lambda tup: tup[1], reverse=True)
                pred_ratings_valid[u] = [r[0] for r in ranked_list_valid[u]]
                pred_ratings__valid[u] = pred_ratings_valid[u][:k]
                p_, r_ = precision_recall(k, pred_ratings__valid[u], validation_matrix[u])
                p_at_.append(p_)
                r_at_.append(r_)

            if abs(np.mean(np.mean(r_at_) - pre_recall) < stop_threshold):
                t_stop_num = t_stop_num + 1
            else:
                t_stop_num = 0
                pre_recall = np.mean(r_at_)
            if t_stop_num > stop_num:
                # performance evaluation based on test set
                pred_ratings_ = {}
                pred_ratings = {}
                pred_score = {}
                pred_score_ = {}
                ranked_list = {}
                a = []
                b = []
                num = - 1
                n_aupr_values = np.zeros([len(unique_users),1])
                for u in unique_users:
                    num += 1
                    user_ids = []
                    user_neg_items = neg_train_matrix[u]
                    item_ids = []

                    for j in user_neg_items:
                        item_ids.append(j)
                        user_ids.append(u)
                    ratings = - self.sess.run([self.pos_distances]
                                              , feed_dict={self.cf_user_input: user_ids,
                                                           self.cf_item_input: item_ids})[0]
                    neg_item_index = list(zip(item_ids, ratings))
                    #print(ratings)
                    #print(ratings.shape)
                    ranked_list[u] = sorted(neg_item_index, key=lambda tup: tup[1], reverse=True)
                    pred_ratings[u] = [r[0] for r in ranked_list[u]]
                    pred_score[u] = [r[1] for r in ranked_list[u]]
                    y_true = []
                    for i in item:
                        if i in test_matrix[u]:
                            y_true.append(1)
                        else:
                            y_true.append(0)

                    print(y_true)
                    # print(len(y_true))
                    print("y_true",np.sum(y_true))
                    y_true = np.array(y_true)[pred_ratings[u]]
                    # print(y_true)
                    # print(y_true.shape)
                    #pred_ratings_[u] = pred_ratings[u][:k]
                    #pred_score_[u] = pred_score[u][:k]

                    #if pred_ratings_.get(u) == None or test_matrix.get(u) == None:continue
                    # p_, r_ = precision_recall(len(item), pred_ratings_[u], test_matrix[u])
                    # print(pred_score_[u][:len(test_matrix[u])])
                    # print(len(pred_score_[u][:len(test_matrix[u])]),type(pred_score_[u][:len(test_matrix[u])][0]))
                    # print(test_matrix[u])
                    # print(len(test_matrix[u]),type(test_matrix[u][0]))
                    print("y_true_new:",y_true)
                    print("y_true_new_sum:",np.sum(y_true))
                    print("pred_score[u]",pred_score[u])
                    precision_r, recall_r, thresholds_r = precision_recall_curve(y_true, pred_score[u])
                    print(precision_r,recall_r)
                    aupr_value = auc(recall_r,precision_r)
                    if aupr_value == np.nan:
                        aupr_value = 0
                    print(aupr_value)
                    n_aupr_values[num] = aupr_value
                    #a.append(p_)
                    #b.append(r_)

                #r_aupr = auc(np.sort(b),np.array(a)[np.argsort(b)])
                r_aupr = np.mean(n_aupr_values)
                for num_k in range(1, 7):
                    k1 = k_Mat[num_k - 1]

                    pred_ratings_ = {}
                    pred_ratings = {}
                    ranked_list = {}
                    p = []
                    r = []
                    for u in unique_users:
                        user_ids = []
                        user_neg_items = neg_train_matrix[u]
                        item_ids = []

                        for j in user_neg_items:
                            item_ids.append(j)
                            user_ids.append(u)
                        ratings = - self.sess.run([self.pos_distances]
                                                  , feed_dict={self.cf_user_input: user_ids,
                                                               self.cf_item_input: item_ids})[0]
                        neg_item_index = list(zip(item_ids, ratings))

                        ranked_list[u] = sorted(neg_item_index, key=lambda tup: tup[1], reverse=True)
                        pred_ratings[u] = [r[0] for r in ranked_list[u]]
                        pred_ratings_[u] = pred_ratings[u][:k1]

                        p_, r_ = precision_recall(k1, pred_ratings_[u], test_matrix[u])
                        p.append(p_)
                        r.append(r_)

                    # compute recall in chunks to utilize speedup provided by Tensorflow
                    r_recalls[:, num_k - 1] = np.mean(np.mean(r))
                    r_precisions[:, num_k - 1] = np.mean(np.mean(p))
                break

        return r_recalls, r_precisions, r_aupr


from load_data import *

if __name__ == '__main__':

    # make feature as dense matrix
    dense_features = None
    Ks_test_recalls = np.zeros([10, 6])
    Ks_test_precisions = np.zeros([10, 6])
    Aupr_values = np.zeros([10, 1])
    df = load_data(path="data/DrDiAssMat2.dat", header=['user_id', 'item_id', 'rating'], sep=" ")
    with tf.Session() as sess:
        train_data, neg_train_matrix, test_data,validation_data ,test_matrix, validation_matrix,num_users, num_items, unique_users,unique_validation= load_ranking_data(df)
        model = MetricFRanking(sess, num_users, num_items, learning_rate=0.01, batch_size=600)
        for num in range(10):
            test_recalls, test_precisions, test_aupr = model.run(train_data, unique_users,unique_validation ,neg_train_matrix,
                                                                 test_matrix, validation_matrix)
            Ks_test_recalls[num, :] = test_recalls
            Ks_test_precisions[num, :] = test_precisions
            Aupr_values[num] = test_aupr
    np.savetxt('CMLDR_recalls.txt', Ks_test_recalls, delimiter='\t')
    np.savetxt('CMLDR_precisions.txt', Ks_test_precisions, delimiter='\t')
    np.savetxt('CMLDR_Aupr_values.txt', Aupr_values, delimiter='\t')
