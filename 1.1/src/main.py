import random
import matplotlib.pyplot as plt
import sys
import vector


def load_data(filename, separator):
    data = []
    file = open(filename)
    for line in file.readlines():
        raw_attr = line.strip().split(separator)
        if len(raw_attr) > 1:
            # REQUIRED: raw_attr的最后一列是标签
            v = vector.Vector([float(raw_attr[i]) for i in range(len(raw_attr) - 1)], raw_attr[len(raw_attr) - 1])
            # REQUIRED: raw_attr的所有列都是数据
            # v = vector.Vector([float(x) for x in raw_attr], '')
            data.append(v)
        # end if
    # end for
    return data


def dist(v1, v2):
    return v1.distTo(v2)


def mean(vectors):
    res = vector.Vector([0 for i in range(len(vectors[0].attr))], '')
    for vec in vectors: res += vec
    attr = [x / len(vectors) for x in res.attr]
    return vector.Vector(attr, '')


def kmeans_show(clusters, x, y, xlabel, ylabel, title):
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
    i = 0
    for key in clusters.keys():
        # 任选两个作为x,y轴
        plt.scatter(key.attr[x], key.attr[y], marker='+', c=colors[i])
        for value in clusters[key]:
            plt.scatter(value.attr[x], value.attr[y], marker='.', c=colors[i])
        # end for
        i = (i + 1) % len(colors)
    # end for
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# 数组A构成n区间: [0,A[0]), [A[0],A[1]), ... , [A[n-2],A[n-1]). x的范围是(0,A[n-1]), 返回x所属的区间编号.
def select_interval(A, x):
    arr = [0] + A

    left, right = 1, len(arr) - 1
    while left <= right:
        i = (left + right) >> 1
        if arr[i - 1] <= x < arr[i]:
            return i - 1
        if x == arr[i]:
            return i
        if x > arr[i]:
            left = i + 1
        else:
            right = i - 1

    assert False
    return -1  # never arrive here


def get_min_dist(u, centroids):
    min_dist = sys.float_info.max
    for v in centroids:
        d = dist(u, v)
        if d < min_dist:
            min_dist = d
    # end for
    return min_dist


# https://www.cnblogs.com/wang2825/articles/8696830.html
def init_centroids(data_set, k):
    centroids = []
    centroids.append(random.choice(data_set))  # 随机选择第一个初始聚类中心

    sum_d = [0] * len(data_set)
    for _ in range(1, k):
        # 计算每个样本与已有聚类中心的最短距离d(即与最近一个聚类中心的距离), 并将其逐项累加到sum_d
        total = 0.0
        for i, u in enumerate(data_set):
            d = get_min_dist(u, centroids)  # 样本u与已有聚类中心之间的最短距离(即与最近一个聚类中心的距离)
            total += d
            if i == 0:
                sum_d[i] = d
            else:
                sum_d[i] = sum_d[i - 1] + d
        # end for
        # 轮盘法选择新的聚类中心
        assert total == sum_d[len(sum_d) - 1]
        i = select_interval(sum_d, total * random.random())
        centroids.append(data_set[i])
    # end for
    return centroids


def k_means(data_set, k, max_iter):
    clusters = {}  # 聚类结果
    mean_vecs = init_centroids(data_set, k)

    for n in range(max_iter):
        # 打印当前的均值向量
        # print('iteration ' + str(n))
        # for vec in mean_vecs: print(vec)
        clusters.clear()
        # 计算data_set中每个样本x与各均值向量vec的距离d, 将x划入距离最近的均值向量closest_vec所对应的聚类
        for x in data_set:
            min_dist = sys.float_info.max
            closest_vec = None
            for vec in mean_vecs:
                d = dist(x, vec)
                if d < min_dist:
                    min_dist = d
                    closest_vec = vec
                # end if
            # end for
            if closest_vec in clusters.keys():
                clusters[closest_vec].append(x)
            else:
                clusters[closest_vec] = [x]
        # end for
        # for循环结束后得到了当前的聚类结果clusters

        # 将dict形式的clusters转换为list形式的tmp_clusters, 便于修改聚类中心
        tmp_clusters = []
        for key in clusters.keys():
            x = list([key])
            x.append(clusters[key])
            tmp_clusters.append(x)

        # 对于tmp_cluster中的每一个cluster, cluster[0]是聚类中心, cluster[1]是一个list——属于聚类中心cluster[0]的全部样本
        changed = False
        for cluster in tmp_clusters:
            old_mean = cluster[0]
            new_mean = mean(cluster[1])
            if old_mean != new_mean:
                cluster[0] = new_mean  # 如果新聚类中心和旧聚类中心不同, 就更新为新的聚类中心
                changed = changed or True
            # end if
        # end for
        if not changed:
            # 聚类中心不再变化, 算法结束
            print('iteration times = %d' % n)
            clusters.clear()
            for cluster in tmp_clusters:
                clusters[cluster[0]] = cluster[1]
            break
        # end if

        # 设置新的均值向量
        mean_vecs.clear()
        for cluster in tmp_clusters: mean_vecs.append(cluster[0])
    # end for
    return clusters


# C1={x,x,x,o,x,o}, 聚类C1中x最多, 把C1看成是x的聚类, 其中x有4个
# C2={x,-,o,-,o,-}, 聚类C2中-最多, 把C2看成是-的聚类, 其中-有3个
# C3={x,x,x,x,x,o}, 聚类C3中x最多, 把C3看成是x的聚类, 其中x有5个
# 准确率 = (4+3+5) / (|C1|+|C2|+|C3|)
def accuracy(clusters):
    correct = 0
    for value in clusters.values():
        stat = {}
        for x in value:
            if x.label in stat.keys():
                stat[x.label] += 1
            else:
                stat[x.label] = 1
        # end for
        stat = dict(zip(stat.values(), stat.keys()))  # key和value互换
        correct += max(stat)  # max(stat)返回的是最大的key
        # key = max(stat)
        # print('%s: %d/%d' % (str(stat[key]), key, len(value)))
    # end for
    # print('total: %d' % sum(len(x) for x in clusters.values()))
    return correct / sum(len(x) for x in clusters.values())


def main():
    data_set = load_data('iris.data', ',')
    clusters = k_means(data_set, 3, 100)
    title = 'k-means (accuracy=%f)' % accuracy(clusters)
    kmeans_show(clusters, 0, 1, 'sepal_length', 'sepal_width', title)
    kmeans_show(clusters, 2, 3, 'petal_length', 'petal_width', title)


if __name__ == '__main__':
    main()
