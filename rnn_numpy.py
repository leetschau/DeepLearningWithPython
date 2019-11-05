import numpy as np

# 下面3行注释针对文本分析的例子，不是本文件代码的含义
timestamps = 100      # 100个时间步，对应100个字符（也可以是单词，但本章用字符作为处理单位）
input_features = 32   # 输入的每个字符是一个长度为32的，由 0, 1 组成的向量
output_features = 64  # 输出的每个单位是一个长度为64的，由 0, 1 组成的向量

inputs = np.random.random((timestamps, input_features))   # inputs.shape: (100, 32)
state_t = np.zeros((output_features, ))

W = np.random.random((output_features, input_features))   # W.shape: (64, 32)
U = np.random.random((output_features, output_features))  # U.shape: (64, 64)
b = np.random.random((output_features, ))                 # b.shape: (64,)

successive_outputs = []
for input_t in inputs:
    # input_t.shape: (32,)
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    # np.dot: 矩阵乘法
    # np.dot(W, input_t).shape: (64,)
    # output_t.shape: (64,)

    successive_outputs.append(output_t)
    state_t = output_t
    # state_t.shape: (64,)

final_output_sequence = np.concatenate(successive_outputs, axis=0)
# final_output_sequence.shape: (6400,)

