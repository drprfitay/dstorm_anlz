import numpy as np
import hdbscan
from sklearn.cluster import DBSCAN
from scipy.spatial import distance_matrix


def random_rotate_3d(pointset):
    # theta = np.random.uniform(0,np.pi*2)
    # rotation_matrix = torch.FloatTensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    # pointset[:,[0,2]] = torch.mm(pointset[:,[0,2]], rotation_matrix)     # random rotation

    # Euler angles
    alpha = np.random.uniform(0, np.pi )
    beta = np.random.uniform(0, np.pi )
    gamma = np.random.uniform(0, np.pi )

    rotation_matrix = np.zeros((3, 3), dtype=np.float)
    rotation_matrix[0,0] = np.cos(beta)*np.cos(gamma)
    rotation_matrix[0,1] = np.cos(gamma)*np.sin(alpha)*np.sin(beta) - np.cos(alpha)*np.sin(gamma)
    rotation_matrix[0,2] = np.cos(alpha)*np.cos(gamma)*np.sin(beta) + np.sin(alpha)*np.sin(gamma)
    rotation_matrix[1,0] = np.cos(beta)*np.sin(gamma)
    rotation_matrix[1,1] = np.sin(alpha)*np.sin(beta)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma)
    rotation_matrix[1,2] = np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma)
    rotation_matrix[2,0] = - np.sin(beta)
    rotation_matrix[2,1] = np.sin(alpha)*np.cos(beta)
    rotation_matrix[2,2] = np.cos(alpha)*np.cos(beta)

    pointset = np.matmul(rotation_matrix, pointset.t()).t()

    return pointset


def cover_space_sample(xyz, radius, max_samples):
    """
    B - Batch, N, npoint - number of points, d-dim coordinates
    Input:
        xyz: pointcloud data, [N, d]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud data, [B, npoint, d]
    """
    N, d = xyz.shape
    grouped = np.zeros(N)
    radius = radius**2

    centroids = np.zeros((max_samples, d), dtype=np.float)
    centroids_idx = np.zeros(max_samples, dtype=np.int64)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N, dtype=np.int64)

    for i in range(max_samples):
        # Sample centroid and calculate all the other points distance to it
        centroids_idx[i] = farthest
        centroids[i] = xyz[farthest, :]
        dist = np.sum((xyz - centroids[i])**2, -1)

        # Find next centroid to sample
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)

        # Check space covering
        mask = dist < radius
        grouped[mask] = 1
        if grouped.all():
            return centroids[:i], centroids_idx[:i]

    print("Not all space were sampled")

    return centroids, centroids_idx


def query_ball_point(radius, xyz, new_xyz):
    """

    Input:
        radius: local region radius
        xyz: all points, [N, d]
        new_xyz: query points, [M, d]
    Return:
        group_idx: grouped points index, [M, nsample]
    """

    # Sampling by nearest neighbors to center
    sqrdists = distance_matrix(new_xyz, xyz, p=2)
    sqrdists[sqrdists > radius] = np.nan
    group_idx = np.argsort(sqrdists, axis=1)
    sqrdists = np.take_along_axis(sqrdists, group_idx, axis=1)

    mask = (sqrdists == sqrdists) # mask nan
    nsample = mask.sum(axis=1).max()

    group_idx = group_idx[:, :nsample]
    mask = mask[:, :nsample]

    return group_idx, mask


def sample_and_group(xyz, radius, max_samples=100, min_npoints=-1):
    """

    Input:
        xyz: input points position data, [N, d]
        radius: local region radius
        nsample: number of sample
        min_npoints: min number of points in group (filter groups with less points)
    Return:
        new_xyz: sampled points position data, [nsample, 3]
        new_points: sampled points data, [nsample, npoint, 3]
    """
    centroids, centroids_idx = cover_space_sample(xyz, radius, max_samples)

    group_idx, mask = query_ball_point(radius, xyz, centroids)
    rows = []
    groups = []
    for row in range(group_idx.shape[0]):
        if mask[row].sum() > min_npoints:
            rows.append(row)
            groups.append(group_idx[row, mask[row]])

    return centroids[rows], groups


def dbscan_cluster_and_group(xyz, 
                             use_hdbscan=False,
                             eps=15, 
                             min_npoints=15, 
                             min_cluster_points=5, 
                             max_std_distance=2.5,
                             hdbscan_min_cluster_points=15,
                             hdbscan_min_samples=1,
                             hdbscan_epsilon_threshold=-9999,
                             hdbscan_extracting_alg="leaf",
                             hdbscan_alpha=1.0):
    """Use HDBSCAN/DBSCAN to cluster a given list of points, then bound them by rects.

        :param xyz: list of lists where each inner list represents a point (row_x, col_y).
        :param eps: int, DBSCAN parameter maximum distance (in pixels) between two points to consider
                            them as a same cluster
        :param min_samples: int, DBSCAN parameter minimum number of adjacent points in the cluster to define
                        a point as core point to cluster.
        :param min_cluster_points: int, minimum number of points to be considered as cluster.
        :param max_std_distance: filter out points in cluster that exceeds a factor of std

       :return:
            new_xyz: sampled points position data, [nsample, 3]
            new_points: sampled points data, [nsample, npoint, 3]
    """

    print("AAAAAlgorithm chosen: %s\n" % ("HDBSCAN" if use_hdbscan else "DBSCAN"))
    try:
        if use_hdbscan:
            if hdbscan_epsilon_threshold != -9999:
                print("Using epsilon threshold: %d" % hdbscan_epsilon_threshold)
                clustering = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_points, 
                                             min_samples=hdbscan_min_samples,
                                             alpha=hdbscan_alpha,
                                             cluster_selection_method=hdbscan_extracting_alg,
                                             cluster_selection_epsilon=hdbscan_epsilon_threshold)
            else: 
                clustering = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_points, 
                                             min_samples=hdbscan_min_samples,
                                             alpha=hdbscan_alpha,
                                             cluster_selection_method=hdbscan_extracting_alg)        
        else:                 
            clustering = DBSCAN(eps=eps, min_samples=min_npoints)

        clustering.fit(xyz)
        #clustering.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
    except BaseException as be:
        print(be)
                 
    print(clustering)
    labels = clustering.labels_
    centroids = []
    groups = []
    for label in np.unique(labels):
        if label != -1:
            cluster = xyz[labels == label]
            distance_from_mean = distance_matrix(cluster, np.expand_dims(cluster.mean(axis=0), axis=0), p=2)[:, 0]
            distance_mean = distance_from_mean.mean()
            distance_std = distance_from_mean.std()
            cluster = cluster[distance_from_mean < (distance_mean + max_std_distance*distance_std)]

            if len(cluster) > min_cluster_points:
                centroids.append(cluster.mean(axis=0))
                groups.append(np.where(labels == label)[0])

    return centroids, groups, np.where(labels == -1)[0]
