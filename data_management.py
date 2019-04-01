# coding=utf-8
__author__ = 'NXG'
import os    # os 处理文件和目录的模块
import cv2
import numpy as np   # 多维数据处理模块
from math import ceil
from random import sample

# 数据集地址
path = 'F:/zns/png/'
# path = 'F:/zns/flower_photos'
data_set_path = '../data_set/'
# 模型保存地址


def mik_dir(path_dir):
    """
    创建目录
    author: NXG    Time:2019/3/15
    :param path_dir: 待创建的目录
    :return: True:创建成功  False：创建失败
    """
    if os.path.isfile(path_dir):
        return False

    def create_direction(not_exists_dir):
        if os.path.exists(not_exists_dir):
            return not_exists_dir
        else:
            not_exists_dir = not_exists_dir.split('/')
            tmp = ''
            for cur_d in not_exists_dir[:]:
                tmp = os.path.join(tmp, cur_d)
                if not os.path.exists(tmp):
                    os.mkdir(tmp)
    if os.path.exists(path_dir):
        return True
    else:
        return create_direction(path_dir)


def read_img(img_path):
    print('img_path yyyyyyyyyy:', img_path)
    img = cv2.imread(img_path)
    print(img.shape)
    # img = cv2.resize(img, (512, 300))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # modify by nxg
    img = img / 255.0
    # img = np.swapaxes(img, 0, 2)
    # img = np.swapaxes(img, 1, 2)     # 1 2 3   3 2 1  3 1 2
    # img = img[np.newaxis, :]
    return img


def shuffle(data,
            labels,
            ratio,
            is_save,
            shuffle_is=True,
            file_name=['data_set', 'label_set', 'test_data_set', 'test_abel_set']):
    """
    打乱数据
    :param data:  待打乱的数据
    :param labels:  待打乱的标签
    :param ratio:  训练集比率
    :param is_save:  [是否将生成的数据集保存到本地，如果本地已有，是否覆盖]
    :param shuffle_is:  是否打乱数据。
    :param file_name:  生成的文件名。
    :return: 打乱的数据,索引
    """
    # 打乱顺序
    # 读取data矩阵的第一维数（图片的个数）
    assert isinstance(is_save, list), 'if you want to save date set is_save must be a list.....'
    num_example = data.shape[0]
    # 训练集
    if shuffle_is:
        index = sample(range(0, num_example, 1), ceil(num_example * ratio))
    else:
        index = list(range(0, num_example, 1))  # 不打乱数据
    # 按照打乱的顺序，重新排序
    train_data = data[index]
    train_label = labels[index]
    # 测试集
    test_index = [s for s in range(num_example) if s not in index]
    if len(test_index) > 0:
        test_data = data[test_index]
        test_label = labels[test_index]

    if is_save[0]:
        if mik_dir(data_set_path):
            print('yes')
            if is_save[-1]:  # data_set, label_set test_data_set test_abel_set
                np.save(file=os.path.join(data_set_path, file_name[0]), arr=train_data, allow_pickle=True, fix_imports=True)
                np.save(file=os.path.join(data_set_path, file_name[1]), arr=train_label, allow_pickle=True, fix_imports=True)
                if len(test_index) > 0:
                    np.save(file=os.path.join(data_set_path, file_name[2]), arr=test_data, allow_pickle=True, fix_imports=True)
                    np.save(file=os.path.join(data_set_path, file_name[3]), arr=test_label, allow_pickle=True, fix_imports=True)
        else:
            assert 'True'.__eq__("False"), 'data set save failed......'
    return data, labels


def create_data_set(path):
    category = [path+x for x in os.listdir(path) if os.path.isdir(path+x)]  # 得到所有的类别目录
    print('cate:', category)
    imgs = []
    labels = []
    for idx, folder in enumerate(category):
        cur_classes_name = os.listdir(folder)
        if len(cur_classes_name) == 0:
            continue
        else:
            for image_name in cur_classes_name:
                # 输出读取的图片的名称
                print('reading the images:%s' % (os.path.join(folder, image_name)))
                img = read_img(os.path.join(folder, image_name))
                imgs.append(img)  # 图片
                labels.append(idx)  # 类别
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


def get_data_set():
    """
    生成数据
    :return:
    """
    data, label = create_data_set(path)  # (1, 100, 100, 3) 0    (633, 100, 100, 3)
    # ['data_set', 'label_set', 'test_data_set', 'test_abel_set']
    # _, _ = shuffle(data, label, .8, [True, True])  # (1, 100, 100, 3) 0    (633, 100, 100, 3)
    _, _ = shuffle(data,
                   label,
                   1.,
                   [True, True],
                   shuffle_is=False,
                   file_name=['triplet_data_set',
                              'triplet_label_set',
                              'triplet_test_data_set',
                              'triplet_test_label_set'])  # (1, 100, 100, 3) 0    (633, 100, 100, 3)


def load_data(saved_data_path, saved_label_path):
    assert os.path.exists(saved_data_path), 'data file path not exists'
    assert os.path.exists(saved_label_path), 'label file path not exists'
    return np.load(saved_data_path), np.load(saved_label_path)


def mini_batch(cur_data, cur_label, shuffle_data=True, batch_size=32):
    """
    获取批量数据
    :param cur_data:  # 数据
    :param cur_label:  # 标签
    :param shuffle_data:  # 默认打乱数据，打乱数据是为了训练，不打乱是为了选择三元组及其它用途
    :param batch_size:  # 批大小为32
    :return:  返回batch 块
    """
    all_mini_batches = []
    all_mini_label = []
    if shuffle_data:
        index_cur = sample(range(len(cur_data)), len(cur_data))
    else:
        index_cur = list(range(len(cur_data)))
    cur_data = cur_data[index_cur]
    cur_label = cur_label[index_cur]
    all_batches = ceil(len(index_cur) / batch_size)
    print('all_batches:', all_batches)
    for mini_index in range(0, all_batches, 1):  # 分批获取数据  1 2 3 4  5 6 7 8    9
        # 避免最后不够batches大小的样本
        next_batch = batch_size * (mini_index + 1)
        if next_batch > len(index_cur):
            remain_index = next_batch - len(index_cur)
            remain_index = index_cur[0: remain_index]
            cur_batch = index_cur[next_batch - batch_size:len(index_cur)] + remain_index
        else:
            cur_batch = index_cur[next_batch - batch_size: next_batch]
        all_mini_batches.append([cur_data[j:j + 1][0] for j in cur_batch])
        all_mini_label.append([cur_label[j:j + 1][0] for j in cur_batch])
    # print('test:', len(all_mini_batches))
    # print('test all_mini_batches[0:', all_mini_batches[0])
    # print('shape all_mini_batches[0:', np.array(all_mini_batches[0]).shape)
    # print('shape all_mini_batches[0:', np.array(all_mini_label[0]).shape)
    return all_mini_batches, all_mini_label

# get_data_set()


def mini_batch_triplet(raw_data_index, shuffle_data=False, batch_size=32):

    """
    获取批量数据
    :param raw_data_index:  # 未打乱的数据样本的索引
    :param shuffle_data:  # 默认打乱数据，打乱数据是为了训练，不打乱是为了选择三元组及其它用途
    :param batch_size:  # 批大小为32
    :return:  返回batch 块
    """
    """
              # # # # # # # # # #     & & & & & & & & & &     * * * * * * * * * * *
              0 0 0 0 0 0 0 0 0 0     1 1 1 1 1 1 1 1 1 1     2 2 2 2 2 2 2 2 2 2 2
              1 2 3 4 5 6 7 8 9 10    11 12 ......
            三元组的选择。两对相同说话人，另外一个为不同的说话人
            
    """
    all_mini_index = dict()
    if shuffle_data:
        raw_data_index = sample(raw_data_index, len(raw_data_index))  # 打乱数据
    all_batches = 0
    for cur_index, cur_data in enumerate(raw_data_index):  # 分批获取数据  1 2 3 4  5 6 7 8    9
        # 避免最后不够batches大小的样本

        if all_mini_index.get(cur_data, -1) == -1:
            all_mini_index[cur_data] = [cur_index]
            all_batches += 1
        else:
            all_mini_index[cur_data].append(cur_index)
    print('all class is：', all_batches)
    return all_mini_index


# get_data_set()
# train_data, train_label = load_data(
#         saved_data_path='../data_set/triplet_data_set.npy',
#         saved_label_path='../data_set/triplet_label_set.npy')
# print('shape of train_data:', train_data.shape)
# print('value of train_label:', train_label)
# print('shape of train_label:', train_label.shape)
#
# index = mini_batch_triplet(train_label)
# print('index:', index)
# data = {0: [0, 0, 0, 0, 0, 0, 0, 0], 1: [1, 1, 1, 1, 1], 2: [2, 2, 2, 2, 2, 2, 2, 2, 2], 3: [3, 3, 3, 3, 3], 4: [4, 4, 4, 4, 4, 4, 4, 4, 4], 5: [5, 5, 5, 5, 5, 5, 5, 5], 6: [6, 6, 6, 6, 6, 6, 6, 6], 7: [7, 7, 7, 7, 7, 7, 7], 8: [8, 8, 8, 8, 8, 8, 8, 8], 9: [9, 9, 9, 9, 9, 9, 9, 9], 10: [10, 10, 10, 10, 10], 11: [11, 11, 11, 11, 11, 11], 12: [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12], 13: [13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13], 14: [14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14], 15: [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], 16: [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16], 17: [17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17], 18: [18, 18, 18, 18, 18, 18, 18, 18, 18], 19: [19, 19, 19, 19, 19, 19, 19, 19, 19], 20: [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20], 21: [21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21], 22: [22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22], 23: [23, 23, 23, 23, 23], 24: [24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24], 25: [25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25], 26: [26, 26, 26, 26, 26, 26, 26, 26], 27: [27, 27, 27, 27, 27, 27], 28: [28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28], 29: [29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29], 30: [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30], 31: [31, 31, 31, 31, 31, 31, 31, 31, 31, 31], 32: [32, 32, 32, 32, 32, 32, 32], 33: [33, 33, 33, 33, 33, 33], 34: [34, 34, 34, 34, 34, 34, 34, 34, 34], 35: [35, 35, 35, 35, 35, 35, 35, 35, 35, 35], 36: [36, 36, 36, 36, 36, 36, 36, 36], 37: [37, 37, 37, 37, 37, 37, 37, 37], 38: [38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38], 39: [39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39], 40: [40, 40, 40, 40, 40], 41: [41, 41, 41, 41, 41], 42: [42, 42, 42, 42, 42, 42, 42], 43: [43, 43, 43, 43], 44: [44, 44, 44, 44], 45: [45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45], 46: [46, 46, 46, 46, 46, 46, 46, 46, 46], 47: [47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47], 48: [48, 48, 48, 48, 48, 48, 48, 48, 48], 49: [49, 49, 49, 49, 49, 49, 49, 49, 49], 50: [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50], 51: [51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51], 52: [52, 52, 52, 52, 52, 52, 52, 52, 52], 53: [53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53], 54: [54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54], 55: [55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55], 56: [56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56], 57: [57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57], 58: [58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58], 59: [59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59], 60: [60, 60, 60, 60, 60, 60], 61: [61, 61, 61, 61, 61, 61, 61], 62: [62, 62, 62, 62, 62, 62, 62, 62, 62], 63: [63, 63, 63, 63, 63], 64: [64, 64, 64, 64, 64, 64, 64], 65: [65, 65, 65, 65, 65, 65, 65], 66: [66, 66, 66, 66, 66, 66, 66, 66, 66, 66], 67: [67, 67, 67, 67, 67, 67, 67], 68: [68, 68, 68, 68, 68, 68, 68], 69: [69, 69, 69, 69, 69, 69, 69, 69, 69, 69], 70: [70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70], 71: [71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71], 72: [72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72], 73: [73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73], 74: [74, 74, 74, 74, 74, 74, 74, 74, 74, 74], 75: [75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75], 76: [76, 76, 76, 76, 76, 76, 76], 77: [77, 77, 77, 77, 77, 77, 77], 78: [78, 78, 78, 78, 78, 78, 78, 78, 78, 78], 79: [79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79], 80: [80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80], 81: [81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81], 82: [82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82], 83: [83, 83, 83, 83, 83, 83, 83, 83], 84: [84, 84, 84, 84, 84], 85: [85, 85, 85, 85, 85], 86: [86, 86, 86, 86, 86], 87: [87, 87, 87, 87, 87, 87], 88: [88, 88, 88, 88, 88, 88, 88, 88, 88], 89: [89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89], 90: [90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90], 91: [91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91], 92: [92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92], 93: [93, 93, 93, 93, 93, 93, 93, 93, 93], 94: [94, 94, 94, 94, 94, 94, 94, 94], 95: [95, 95, 95, 95, 95, 95, 95, 95], 96: [96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96], 97: [97, 97, 97, 97, 97, 97, 97], 98: [98, 98, 98, 98, 98, 98, 98], 99: [99, 99, 99, 99, 99, 99, 99, 99, 99]}
# keys_data = data.keys()
# sample_triplet = sample(keys_data, 64)
# pair_data = sample_triplet[:32]
# signal_data = sample_triplet[32:64]
#
# print(sample_triplet)
# print(pair_data)
# print(signal_data)
