## Metric Factorization: Recommendation beyond Matrix Factorization



### 数据的分割与采样：  loaddata.py 文件的load_ranking_data(path="data/DrDiAssMat2.dat", test_size=0.2, header=['user_id', 'item_id', 'rating'], sep=" ")

1. 传入的数据格式为 

   | user_id | item_id | rating |
   | ------- | ------- | ------ |
   | 1       | 504     | 1      |
   | 2       | 10      | 1      |
   | 2       | 32      | 1      |
   | 2       | 67      | 1      |
   | 2       | 189     | 1      |
   | 2       | 443     | 1      |
   | .....   | .....   | ...... |

2. 函数的返回值

   ```python
   # train_martix.todok(),返回每行为 (row_index,column_index) train_value(1 || 0) 的数据
   # neg_user_item_matrix,返回的为字典形式 {key:value} key为user_id value为没有出现在train_data中的item_id
   # test_matrix.todok() 返回每行为 (row_index,column_index) train_value(1) 的数据
   # test_user_item_matrix 返回的为字典形式 {key:value} key为user_id value为出现在test_data中的item_id
   # n_user user_id的总数量
   # n_items item_id的总数量
   # set(unique_user) 测试集的user_id 的数量
   return train_matrix.todok(),  neg_user_item_matrix, test_matrix.todok(), test_user_item_matrix, 		n_users, n_items, set(unique_users)
   ```



### 主程序： MerticFRanking.py 文件中的MerticFRanking 类

1. 初始化模型

   ```python
   train_data, neg_train_matrix, test_data, test_matrix, num_users, num_items, unique_users \
           = load_ranking_data(path="data/DrDiAssMat2.dat", header=['user_id', 'item_id', 'rating'], sep=" ")
   model = MetricFRanking(sess,num_users,num_items)
   ```

2. 参数的设置

   ```python
   # 学习率 self.learn = 0.1
   # 训练循环次数 epoch = 200
   # 隐向量 N = 100
   # 用户的数量 num_users
   # 商品数量 num_items 
   # 训练一次模型传入的数据量 batch_size = 1024 * 3
   # 正则化项 clip_norm = 1 L2正则项会将所有的点向坐标轴推，这个可以放松正则化要求
   # beaa = 1.5
   ```

3. 模型训练

   ```python
       def run(self, train_data, unique_users, neg_train_matrix, test_matrix):
           # train_data: 训练数据
           # unique_users: 用户ID列表
           # neg_train_matrix: 负面训练矩阵
           # test_matrix: 验证数据
           
           # 初始化变量 ########################################################################## 
            self.cf_user_input = tf.placeholder(dtype=tf.int32, shape=[None], name='cf_user_input')
           self.cf_item_input = tf.placeholder(dtype=tf.int32, shape=[None], name='cf_item_input')
           # self.y = tf.placeholder("float", [None], 'y')
           self.y = tf.placeholder(dtype=float, shape=[None], name='y')
           # m * N 的矩阵，初始化用户潜在因子矩阵
           U = tf.Variable(tf.random_normal([self.num_users, self.N], stddev=1 / (self.N ** 0.5)), dtype=tf.float32)
           # n * N 的矩阵，初始化物品潜在因子矩阵
           V = tf.Variable(tf.random_normal([self.num_items, self.N], stddev=1 / (self.N ** 0.5)), dtype=tf.float32)
           ######################################################################################
           
           # 进行函数的映射，可以取得映射的user_id , item_id ###################################3
           # 用户ID和对应的嵌入数据（潜在的用户向量）映射
           users = tf.nn.embedding_lookup(U, self.cf_user_input)
           # 物品ID和对应的嵌入数据（潜在的物品向量）映射
           pos_items = tf.nn.embedding_lookup(V, self.cf_item_input)
           ######################################################################################
           
              # 获取对应的用户和物品距离差值
           self.pos_distances = tf.reduce_sum(tf.squared_difference(users, pos_items), 1, name="pos_distances")
           # dropout ，用于优化关于神经网络的过拟合的问题
           self.pred = tf.reduce_sum(tf.nn.dropout(tf.squared_difference(users, pos_items), 0.95), 1, name="pred")
   
           # 物品排序的损失函数
           self.loss = tf.reduce_sum((1 + 0.1 * self.y) * tf.square(
               (self.y * self.pred + (1 - self.y) * tf.nn.relu(self.beta * (1 - self.y) - self.pred))))
           
           # 这里使用AdagradOptimizer可以自动的调节leaning_rating
           self.optimizer = tf.train.AdagradOptimizer(self.lr).minimize(self.loss, var_list=[U, V])
            # 这里的clip_by_norm是指对梯度进行裁剪，通过控制梯度的最大范式，防止梯度爆炸的问题，是一种比较常用的梯度规约的方式。
           clip_U = tf.assign(U, tf.clip_by_norm(U, self.clip_norm, axes=[1]))
           clip_V = tf.assign(V, tf.clip_by_norm(V, self.clip_norm, axes=[1]))
           
           # 将压缩稀疏矩阵转化为坐标矩阵，其中的坐标矩阵存储着(行，列，值)
           temp = train_data.tocoo()
           # 获取train_item_id
           item = list(temp.col.reshape(-1))
           # 获取train_user_id
           user = list(temp.row.reshape(-1))
           # 获取train_rating
           rating = list(temp.data)
           
           # 这里的神经网络训练为epoches = 200
           for epoch in range(self.epochs):
               user_temp = user[:]
               item_temp = item[:]
               rating_temp = rating[:]
               
               ##从未评分的数据中采用负例 ######################################################### 
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
               item_temp += item_append
               user_temp += user_append
               rating_temp += values_append
               self.num_training = len(rating_temp)
               total_batch = int(self.num_training / self.batch_size)
               ################################################################################
   
               # 随机排序列表
               idxs = np.random.permutation(self.num_training)
               user_random = list(np.array(user_temp)[idxs])
               item_random = list(np.array(item_temp)[idxs])
               rating_random = list(np.array(rating_temp)[idxs])
   			
               # 这里使用的分批输入数据进行训练
               for i in range(total_batch):
                   batch_user = user_random[i * self.batch_size:(i + 1) * self.batch_size]
                   batch_item = item_random[i * self.batch_size:(i + 1) * self.batch_size]
                   batch_rating = rating_random[i * self.batch_size:(i + 1) * self.batch_size]
   
                   _, c, p, _, _ = self.sess.run((self.optimizer, self.loss, self.pos_distances, clip_U, clip_V), feed_dict={self.cf_user_input: batch_user, self.cf_item_input: batch_item,self.y: batch_rating})
               	avg_cost = c
   			
               # 这里神经网络虽然训练了200次，但是输出只有100次
               if (epoch) % 2 == 0 and epoch >= 0:  
   
                   pred_ratings_10 = {}
                   pred_ratings_50 = {}
                   pred_ratings = {}
                   ranked_list = {}
                   count = 0
                   p_at_50 = []
                   r_at_50 = []
                   for u in unique_users:
                       user_ids = []
                       count += 1
                       user_neg_items = neg_train_matrix[u]
                       item_ids = []
   
                       for j in user_neg_items:
                           item_ids.append(j)
                           user_ids.append(u)
                           # 这里要根据距离对item进行排序，这里取负值的目的是因为距离越远，优先级越低
                       ratings = - self.sess.run([self.pos_distances] , feed_dict={self.cf_user_input: user_ids, self.cf_item_input: item_ids})[0]
                       
                       neg_item_index = list(zip(item_ids, ratings))
                       ranked_list[u] = sorted(neg_item_index, key=lambda tup: tup[1], reverse=True)
                       pred_ratings[u] = [r[0] for r in ranked_list[u]]
                       
                       # 这里50是决定是取前多少个item
                       pred_ratings_50[u] = pred_ratings[u][:10]
   				
                   	# 这里返回p_at_50,r_at_50是针对一个用户的精确度和召回率
                       p_50, r_50 = precision_recall_ndcg_at_k(10, pred_ratings_50[u], test_matrix[u])
                       p_at_50.append(p_50)
                       r_at_50.append(r_50)
                   print("-------------------------------")
                   print("precision@50:" + str(np.mean(p_at_50)))
                   MF_precisions.append(np.mean(p_at_50))
                   print("recall@50:" + str(np.mean(r_at_50)))
                   MF_recalls.append(np.mean(r_at_50))
                   MF_Aupr_vaules.append(auc(np.sort(np.array(r_at_50)),np.array(p_at_50)[np.argsort(r_at_50)]))
                   print("auc",auc(np.sort(np.array(r_at_50)),np.array(p_at_50)[np.argsort(r_at_50)]))
                   
           # 这里是将100个数据储存起来
           np.savetxt("MF_recalls.txt",MF_recalls,delimiter="\t")
           np.savetxt("MF_precisions.txt", MF_precisions, delimiter="\t")
           np.savetxt("MF_Aupr_values.txt", MF_Aupr_vaules, delimiter="\t")
           
           
   ```

   ### 关于评价指标

   可以计算精确度和召回率以及auc的值，具体计算可以见函数precision_recall_ndcg_at_k,以及根据precision和recall计算auc的值

   ```python
   def precision_recall_ndcg_at_k(k, rankedlist, test_matrix):
   	# 这里的k为是决定是取前多少个item
       b1 = rankedlist
       b2 = test_matrix
       s2 = set(b2)
       hits = [ (idx, val) for idx, val in enumerate(b1) if val in s2]
       count = len(hits)
       return float(count / k), float(count / len(test_matrix))
   ```

   

   程序会生成三个txt文件，文件分别为MF_recalls.txt，MF_precisions.txt，MF_Aupr_values.txt

   MF_recalls.txt ：文件共有100行，每一行表示，模型每训练两次后，进行测试在测试集上的各个user召回率的平均值

   MF_precisions.txt ： 文件也共有100行 每一行表示，模型每训练两次后，进行测试在测试集上的各个user精确率的平均值

   MF_Aupr_values.txt ：文件也共有100行 每一行表示，模型每训练两次后，进行测试在测试集上的各个user上auc的平均值