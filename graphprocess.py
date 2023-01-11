import sys
import pickle as pkl
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.metrics import pairwise_distances as pair
# from utils import load__data, DataSet
import torch
import scipy.io as scio


def data_partition_random(dataset_dir, dataset_name, label_n_per_class):
    print('加载的数据集是：{}'.format(dataset_name))
    print('每个标签给了{}个'.format(label_n_per_class))
    # 随机数据分区
    text_set_n = 1000
    val_set_n = 500
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, one_hot_labels = load__data(dataset_name,
                                                                                                           dataset_dir)

    # baocunshuju(adj)
    n = len(y_train)  # 2708
    k = len(y_train[0])  # 7

    labels = one_hot_labels.argmax(axis=1)  # 所有节点的标签

    train_index_new = np.zeros(k*label_n_per_class).astype(int)  # 训练集的标签

    train_mask_new = np.zeros(n).astype(bool)  # 训练集标识
    val_mask_new = np.zeros(n).astype(bool)  # 验证集标识
    test_mask_new = np.zeros(n).astype(bool)  # 测试集标识

    y_train_new = np.zeros((n, k))  # 训练集二维数组,规模：训练集个数*训练集类别个数。初始化全0
    y_val_new = np.zeros((n, k))  # 验证集
    y_test_new = np.zeros((n, k))  # 测试集

    class_index_dict = {}
    for i in range(k):
        class_index_dict[i] = np.where(labels == i)[0]  # 将各个类别(0-6)对应的下标按类别分类

    for i in range(k):  # 每个类别只给label_n_per_class（10）个标签
        class_index = class_index_dict[i]
        train_index_one_class = np.random.choice(class_index, label_n_per_class, replace=False)  # 从class_index(一维)中随机抽取数字，并组成label_n_per_class大小(10)的数组,replace:False不可以取相同数字
        train_index_new[i*label_n_per_class:i*label_n_per_class + label_n_per_class] = train_index_one_class  # 放在数组相应的位置上

    train_index_new = list(train_index_new)
    test_val_potential_index = list(set([i for i in range(n)]) - set(train_index_new))  # 测试集潜在的序列 ： 所有数据-训练数据
    test_index_new = np.random.choice(test_val_potential_index, text_set_n, replace=False)  # 从潜在测试集中随机选1000个
    potential_val_index = list(set(test_val_potential_index) - set(test_index_new))  # 潜在的验证集的序列 ： 所有测试集-选中的测试集
    val_index_new = np.random.choice(potential_val_index, val_set_n, replace=False)  # 从潜在的验证集中随机选出500个

    train_mask_new[train_index_new] = True  # 训练集的标识(本来都是false,在训练集所对应序列换成true)
    val_mask_new[val_index_new] = True  # 验证集的标识
    test_mask_new[test_index_new] = True  # 测试集的标识

    for i in train_index_new:
        y_train_new[i][labels[i]] = 1  # 被选中的训练集的二维列表,行是某训练集对应的序列号，列是某训练集对应的类别。

    for i in val_index_new:
        y_val_new[i][labels[i]] = 1  # 被选中的验证集

    for i in test_index_new:
        y_test_new[i][labels[i]] = 1  # 测试集

    return adj, features, y_train_new, y_val_new, y_test_new, train_mask_new, val_mask_new, test_mask_new, one_hot_labels


def load_data__my(g_adj, g_feature, g_lable, g_train, g_val, g_test):
    """Load citation network dataset (cora only for now)"""
    print('Loading DataSet...')
    features = g_feature

    adj = g_adj
    b = 0
    train = []
    for i in g_train:
        if i:
            train.append(b)
        b += 1
    b = 0
    val = []
    for i in g_val:
        if i:
            val.append(b)
        b += 1
    b = 0
    test = []
    for i in g_test:
        if i:
            test.append(b)
        b += 1
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(g_lable)[1])  # cora用的
    idx_train = torch.LongTensor(train)
    idx_val = torch.LongTensor(val)
    idx_test = torch.LongTensor(test)
    # return adj, features, labels, idx_train, idx_val, idx_test
    return DataSet(x=features, y=labels, idx_train=idx_train, idx_test=idx_test, mask_train=torch.FloatTensor(g_train),
                   mask_test=torch.FloatTensor(g_test), homophilys=0.0, homophilyf=0.0, adj=adj)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def process_data(dataset):
    names = ['y', 'ty', 'ally', 'x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("../data/cache/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    y, ty, ally, x, tx, allx, graph = tuple(objects)
    print(graph)
    test_idx_reorder = parse_index_file("../data/cache/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = features.toarray()
    print(features)
    f = open('../data/{}/{}.adj'.format(dataset, dataset), 'w+')
    for i in range(len(graph)):
        adj_list = graph[i]
        for adj in adj_list:
            f.write(str(i) + '\t' + str(adj) + '\n')
    f.close()

    label_list = []
    for i in labels:
        label = np.where(i == np.max(i))[0][0]
        label_list.append(label)
    np.savetxt('../data/{}/{}.label'.format(dataset, dataset), np.array(label_list), fmt='%d')
    np.savetxt('../data/{}/{}.test'.format(dataset, dataset), np.array(test_idx_range), fmt='%d')
    np.savetxt('../data/{}/{}.feature'.format(dataset, dataset), features, fmt='%f')


def construct_graph(dataset, features, topk):
    # fname = '../data/' + dataset + '/knn/tmp.txt'
    # fname = '../data/' + dataset + '/knn/tmp.txt'
    # fname = '../data/2V_MNIST_USPS/knn/tmp.txt'
    datasetname = dataset[:-4]
    fname = 'data/{}/knn/tmp.txt'.format(datasetname)
    print(fname)
    f = open(fname, 'w')
    ##### Kernel
    # dist = -0.5 * pair(features) ** 2
    # dist = np.exp(dist)

    #### Cosine
    dist = cos(features)
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)

    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass
            else:
                f.write('{} {}\n'.format(i, vv))
    f.close()


def generate_knn(dataset):
    for topk in range(2, 10):
        if dataset != 'cora' and dataset != 'pubmed':
            # data = np.loadtxt('../data/' + dataset + '/' + dataset + '.feature', dtype=float)
            data = scio.loadmat("Caltech-5V.mat")
            x5 = data['X5']

        else:
            data = load_data(dataset)
        # print(data)
        # construct_graph(dataset, data, topk)
        #此处若视图不是二维的则要转成二维的
        # x1 = x1.reshape((5000, -1))
        construct_graph(dataset,x5,topk)

        print(x5.shape)
        # f1 = open('../data/' + dataset + '/knn/tmp.txt', 'r')
        #f2 = open('../data/' + dataset + '/knn/c' + str(topk) + '.txt', 'w')
        # f1 = open('../data/2V_MNIST_USPS/knn/tmp.txt', 'r')
        # f2 = open('../data/2V_MNIST_USPS/knn/c' + str(topk) + '.txt', 'w')
        datasetname = dataset[:-4]
        f1 = open(f'data/{datasetname}/knn/tmp.txt', 'r'.format(datasetname=datasetname))
        # f2 = open('data/2V_MNIST_USPS/knn/x1_' + str(topk) + '.txt', 'w')
        file_path2 = 'data/{dataset}/knn/{xi}_{topk}.txt'.format(dataset=datasetname, xi='x5', topk=topk)
        f2 = open(file_path2, 'w')
        lines = f1.readlines()
        for line in lines:
            start, end = line.strip('\n').split(' ')
            if int(start) < int(end):
                f2.write('{} {}\n'.format(start, end))
        f2.close()


def load_data(dataset):
    adj_gcn, node_features, y_train, y_val, y_test, train_mask, val_mask, test_mask, one_hot_labels = data_partition_random(
        dataset_dir='../data/' + dataset + '/', dataset_name=dataset, label_n_per_class=20)
    data = load_data__my(adj_gcn, node_features, one_hot_labels, train_mask, val_mask, test_mask)
    return data.x.cpu().numpy()


''' process cora/citeseer/pubmed data '''
# process_data('citeseer')

'''generate KNN graph'''

# generate_knn('pubmed')
generate_knn('Caltech-5V.mat')