# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/19 21:55'

import pandas as pd
import numpy as np

data = pd.read_table('DrDiAssMat.txt', header=None)
data = np.array(data)
with open("DrDiAssMat3.dat", 'w') as f:
    for row, line in enumerate(data, start=1):
        for column, one in enumerate(line, start=1):
            if one == 1:
                f.write(str(row) + " " + str(column) + " " + str(1) + "\n")
