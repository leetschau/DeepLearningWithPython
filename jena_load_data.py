"""读取源数据，定义数据加载函数

"""

import numpy as np

def data_gen(data, lookback, delay, min_index, max_index,
        shuffle=False, batch_size=128, step=6):
    """生成历史和预测数据集

    算法执行过程：首先在合理区间里找出一系列时间基准值保存在 rows 中，
    根据 shuffle 的取值以及上次取值结束位置（保存在 i 中），
    这个合理区间可能是：(min_index + lookback, max_index)，
    或者 (i, i + batch_size), (i, max_index)。

    对 rows 的每个元素 row，以 row - lookback 为左边界、
    row 为右边界、step 为间隔，生成一系列历史数据采样点，
    将这些时间点上所有 14 个测量值被放入 samples 中，
    将 row + delay 时间点上的温度值作为标签（预测值，与算法给出的预测结果比较）
    放入 target 里。

    本例中，一个批量取 128 个样本，一个样本有一个基准时间，
    向前推10天，在这个区间内等间隔取240个点（每1小时采样一次），
    每个点上取源数据中所有14个物理测量值，形成一个 240 x 14 的矩阵作为预测依据，
    再取基准时间后1天的温度作为预测目标。

    Parameters:
    data (numpy.ndarray): 本例中为 420551 行，14 列
    lookback (int): 历史数据长度，本例中为 1440，即 10 天（原始数据间隔为10分钟）；
    delay: 从基准时间（lookback 的结束时间点）向后推 delay 时间，预测目标是这个时间点上的温度值
    min_index (int): 历史区间的左边界
    max_index (int): 历史区间的右边界
    batch_size (int): 样本个数
    step: 采样点时间间隔，本例中 6 表示每 6 个点采样一次，即采用间隔为 1 小时

    Returns:
    tuple: 包含（历史 预测）二元组，
    第一部分是形状为 (batch_size, lookback/step, feature_number)，
    本例中为 (128, 240, 14) 的 numpy.ndarray，
    第二部分（预测）是一个长度为 batch_size 的一元 ndarray，
    本例中形状为 (128,)。
    """
    if max_index is None:
        max_index = len(data) - delay - 1  # 防止 target 取值时数组下标右越界
    i = min_index + lookback   # 防止 samples 向回取历史值（保存在 indices 里）时左越界
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback,
                                     max_index, size=batch_size)
        # rows 的每个元素以自己为右边界，生成一个时间序列作为历史和一个预测值，
        # 彼此之间没有顺序，所以可以乱序。
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))  # data 的最后一个维度，即除时间戳外的特征数，14
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets


fname = './jena_climate_2009_2016.csv'

with open(fname, 'r') as f:
    rawlines = f.readlines()

striped = map(lambda x: x.strip(), rawlines)
header = next(striped).split(',')
lines = list(striped)
print('Header:', header)
print('Number of data lines:', len(lines))

float_data = np.zeros((len(lines), len(header) - 1))  # 在 header 包含的列数上 -1 是因为不包含时间列
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i] = values

mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std
