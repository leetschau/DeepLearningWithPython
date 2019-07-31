# explanations of imdb_cls.py

import numpy as np
from imdb_cls import train_data

results = np.zeros((len(train_data), 10000))
results[0, [3,4,5,4,2]] = 3
# 可以通过 list 给 numpy.ndarray 的某个维度的多个元素一次性赋值，
# 且 list 中可以包含重复的值，例如上面 [3,4,5,4,2] 中的 4，
# 但最终的效果和 [2,3,4,5] 一样

print(results[0:5, 0:5])

train_enum = enumerate(train_data)
i, sequence = next(train_enum)
assert sequence == train_data[0]

results[0,:] = 0 # 回退到初始状态

results[i, sequence] = 1
# 将第一条评论的所有单词 index 设置为1，丢失了单词的顺序和次数信息
# 例如 "this movie is so so so good" 和 "so good this movie" 被 `vectorize_sequences`
# 处理后的结果应该是一样的
