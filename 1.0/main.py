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


def kmeans_show(clusters, x, y, xlabel, ylabel):
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
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def k_means(data_set, k, max_iter):
    clusters = {}  # 聚类结果
    # mean_vecs = random.sample(data_set, k)  # 随机选择k个样本作为初始均值向量
    mean_vecs = data_set[:k]  # 选择前k个样本作为均值向量

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
            print('iteration times = %d. break' % n)
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


def main():
    data_set = load_data('iris.data', ',')
    clusters = k_means(data_set, 3, 100)
    kmeans_show(clusters, 0, 1, 'sepal_length', 'sepal_width')
    kmeans_show(clusters, 2, 3, 'petal_length', 'petal_width')


if __name__ == '__main__':
    main()
