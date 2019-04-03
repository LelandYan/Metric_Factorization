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

   数据表示的意义为每行的user_id 对 item_id 感兴趣

   ### 与原程序的输入数据相同

2. 读入数据文件load_data.py

   对原文件loaddata.py进行了改写

   1. 进一步对程序进行解耦

      使用3个函数分别执行对数据的读入、训练集稀疏矩阵的生成、以及测试集和验证集的稀疏矩阵的生成

      1. 函数load_data 对数据的读入
      2. load_randking_data 训练集稀疏矩阵的生成
      3. form_csr_matrix 测试集和验证集的稀疏矩阵的生成

   2. 将分割数据集的方式由分为训练集和测试集改为训练集、测试集、验证集 分割比例为6:2:2

   3. 将原程序中的计算n_users和n_items由去对原数据中的n_users和n_items去重后去个数更改为取n_users和n_items的最大序号，这里的修改是因为程序构建矩阵的方式是通过将矩阵的索引和n_users和n_items的序号相联系，也就是通过矩阵的索引来寻找n_users和n_items的值

   4. 对于函数load_ranking_data的返回值做了修改

      1. 原程序函数的返回值为 

         train_matrix.todok() 训练集所对应的稀疏矩阵

         neg_user_item_matrix  训练集所对应的负采样字典

         test_matrix.todok() 测试集所对应的稀疏矩阵 --- 这里的与训练集不相同，n_users和n_items并不是全部的，只是测试集所对应的n_users和n_itmes

         test_user_item_matrix 获取测试集中的已知的n_users和n_items的关系

         n_users, 获取全部的n_users

         n_items 获取全部的n_items

         set(unique_users)  这里获取的是 测试集中的由多少不重复的n_users

      2. 现程序函数的返回值

         train_matrix.todok() 训练集所对应的稀疏矩阵

         neg_user_item_matrix  训练集所对应的负采样字典

         test_matrix.todok() 测试集所对应的稀疏矩阵 --- 这里的与训练集不相同，n_users和n_items并不是全部的，只是测试集所对应的n_users和n_itmes

         validation_matrix.todok() 验证集对应的稀疏矩阵 -- 这里的与训练集不相同，n_users和n_items并不是全部的，只是验证集所对应的n_users和n_itmes

         test_user_item_matrix 获取测试集中的已知的n_users和n_items的关系

         validation_user_item_matrix 获取验证集中已知的n_users和n_items的关系

         n_users, 获取全部的n_users

         n_items 获取全部的n_items

         set(unique_users)  这里获取的是 测试集中的由多少不重复的n_users

         set(unique_users_validation) 这里是获取 验证集中有多少不重复的n_users

3. 

   

   

   

   

   

   

   

   

   

   