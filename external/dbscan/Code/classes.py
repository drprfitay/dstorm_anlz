"""
****---------------------------CLASS FILE---------------------------****
"""

import numpy as np
from matplotlib.mlab import PCA
import csv
import math


def dist(p1, p2, d3=True):
    a = (p1[0] - p2[0]) ** 2
    b = (p1[1] - p2[1]) ** 2
    c = (p1[2] - p2[2]) ** 2

    if d3 == True:
        return math.sqrt(a + b + c)
    else:
        return math.sqrt(a + b)


def dotproduct(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def angle(v1, v2):
    return math.acos(dotproduct(v1, v2)) * 57.296


def princomp(A):
    """ performs principal components analysis
        (PCA) on the n-by-p data matrix A
        Rows of A correspond to observations, columns to variables.

    Returns :
     coeff :
       is a p-by-p matrix, each column containing coefficients
       for one principal component.
     score :
       the principal component scores; that is, the representation
       of A in the principal component space. Rows of SCORE
       correspond to observations, columns to components.
     latent :
       a vector containing the eigenvalues
       of the covariance matrix of A.
    """
    # computing eigenvalues and eigenvectors of covariance matrix
    M = (A - np.mean(A.T, axis=1)).T  # subtract the mean (along columns)
    [latent, coeff] = np.linalg.eig(np.cov(M))  # attention:not always sorted
    score = np.dot(coeff.T, M)  # projection of the data in the new space
    return coeff, score, latent


def to_point(p):  # makes a Point out of point.
    return Point(x=p[0], y=p[1], z=p[2])


def to_np(arr):  # transforms an array to np array
    return np.array([point.point for point in arr], dtype=np.float)


# ***-----------------------POINT----------------------------------***#

class Point:
    def __init__(self, x=0, y=0, z=0, visited=False, isnoise=False,
                 intensity=1, color=None):
        self.point = np.array([x, y, z], dtype=np.float)
        self.intensity = 1  # N/A
        self.cluster = -1
        self.p2cluster = -1
        self.green_cluster = -1
        self.red_cluster = -1
        self.color = color
        self.distance_from_center = 0
        self.opposite_clusters = 0


# ***-----------------------SAMPLE----------------------------------***#

class Sample:
    def __init__(self, green_path, red_path, epsilon=90, min_n=8, path="D:/Gilad/pyxy", data_type="3d", name=""):

        self.name = name
        self.path = path
        self.green_path = green_path
        self.red_path = red_path
        self.f = ""
        self.f_clusters = ""
        self.f_clusters_final = ""
        self.f_clusters_pre = ""
        self.f_clusters_all = ""
        self.f_csv = ""
        self.points = []
        self.green_points = []
        self.red_points = []
        self.points_dbscan = []
        self.green_points_dbscan = []
        self.red_points_dbscan = []
        self.intensities = []
        self.clustered_points = []
        self.unclustered_points = []
        self.all_clustered_points = []
        self.all_unclustered_points = []
        self.points_size = 0
        self.green_size = 0
        self.red_size = 0
        self.data_type = data_type
        self.part2_points = []
        self.part2_points_dbscan = []

        self.part2clusters = []
        self.clusters = []
        self.clusters_green = []
        self.clusters_red = []
        self.green_clusters = []
        self.red_clusters = []
        self.pre_red_clusters = []
        self.pre_green_clusters = []
        self.clusters_size = 0
        self.labels = []
        self.green_labels = []
        self.red_labels = []
        self.centers = []
        self.green_centers = []
        self.red_centers = []

        self.epsilon = epsilon
        self.min_n = min_n
        self.mini_eps = 50
        self.mini_minimum_ngbs = 8
        self.parse = ""

    def get_points(self, which="3d"):

        self.parse = which

        if which == "3d":
            return self.parse_3d(16, 17, 18)
        elif which == "new_3d":
            return self.parse_3d(15, 16, 17)
        elif which == "2d":
            return self.parse_2d(16, 17)
        elif which == "new_2d":
            return self.parse_2d(15, 16)
        elif which == "raw_3d":
            return self.parse_raw_3d()
        elif which == "raw_2d":
            return self.parse_raw_2d()
        elif which == "old":
            return self.parse_old()

        else:
            print("invalid parse selection. argument should be \"3d\"\
            , \"2d\", \"raw_3d\", \"raw_2d\" or \"old\" or \"new_3d\" or \"new_2d\"")

    def parse_3d(self, colx, coly, colz):
        with open(self.green_path) as g:
            g = csv.reader(g, delimiter=',')
            cnt = 0
            g_cnt = 0
            for row in g:
                if (cnt != 0):
                    point = Point(x=row[colx], y=row[coly], z=row[colz],
                                  color="green")
                    if float(point.point[2]) < 600 or float(point.point[2]) > -600:
                        self.points.append(point)
                        self.green_points.append(point)
                        self.green_points_dbscan.append(point.point)
                        self.points_dbscan.append(point.point)
                        g_cnt += 1
                cnt += 1

            self.green_size = g_cnt

        with open(self.red_path) as r:
            r = csv.reader(r, delimiter=',')
            r_cnt = 0
            cnt = 0
            for row in r:
                if (cnt != 0):
                    point_r = Point(x=row[colx], y=row[coly], z=row[colz],
                                    color="red")
                    if float(point_r.point[2]) < 600 or float(point_r.point[2]) > -600:
                        self.points.append(point_r)
                        self.red_points.append(point_r)
                        self.red_points_dbscan.append(point_r.point)
                        self.points_dbscan.append(point_r.point)
                    r_cnt += 1
                cnt += 1

            self.red_size = r_cnt

        return

    def parse_old(self):
        with open(self.green_path) as g:
            g = csv.reader(g, delimiter=' ')
            cnt = 0
            g_cnt = 0
            r_cnt = 0
            for row in g:
                if (cnt != 0 and cnt < 200):
                    point = Point(x=str(float(row[0]) * 100), y=str(float(row[1]) * 100), z='0',
                                  color="green")
                    self.points.append(point)
                    self.green_points.append(point)
                    self.green_points_dbscan.append(point.point)
                    self.points_dbscan.append(point.point)
                    g_cnt += 1
                else:
                    point = Point(x=str(float(row[0]) * 100), y=str(float(row[1]) * 100), z='0',
                                  color="red")
                    #                        if float(point.point[2]) < 600 or float(point.point[2]) > -600:
                    self.points.append(point)
                    self.red_points.append(point)
                    self.red_points_dbscan.append(point.point)
                    self.points_dbscan.append(point.point)
                    r_cnt += 1
                cnt += 1

            self.green_size = g_cnt
            self.red_size = r_cnt

        return

    def parse_raw_3d(self):
        with open(self.green_path) as g:
            g = csv.reader(g, delimiter=',')
            cnt = 0
            g_cnt = 0
            r_cnt = 0
            for row in g:
                if (cnt != 0):
                    if row[6] == '0':
                        point = Point(x=row[16], y=row[17], z=row[18],
                                      color="green")
                        if float(point.point[2]) < 600 or float(point.point[2]) > -600:
                            self.points.append(point)
                            self.green_points.append(point)
                            self.green_points_dbscan.append(point.point)
                            self.points_dbscan.append(point.point)
                            g_cnt += 1
                    else:
                        point_r = Point(x=row[16], y=row[17], z=row[18],
                                        color="red")
                        if float(point_r.point[2]) < 600 or float(point_r.point[2]) > -600:
                            self.points.append(point_r)
                            self.red_points.append(point_r)
                            self.red_points_dbscan.append(point_r.point)
                            self.points_dbscan.append(point_r.point)
                            r_cnt += 1
                cnt += 1

            self.green_size = g_cnt
            self.red_size = r_cnt

        return

    def parse_2d(self, colx, coly):
        with open(self.green_path) as g:
            g = csv.reader(g, delimiter=',')
            cnt = 0
            g_cnt = 0
            for row in g:
                if (cnt != 0):
                    point = Point(x=row[colx], y=row[coly], z='0',
                                  color="green")
                    if float(point.point[2]) < 600 or float(point.point[2]) > -600:
                        self.points.append(point)
                        self.green_points.append(point)
                        self.green_points_dbscan.append(point.point[:2])
                        self.points_dbscan.append(point.point[:2])
                        g_cnt += 1
                cnt += 1

            self.green_size = g_cnt

        with open(self.red_path) as r:
            r = csv.reader(r, delimiter=',')
            r_cnt = 0
            cnt = 0
            for row in r:
                if (cnt != 0):
                    point_r = Point(x=row[colx], y=row[coly], z='0',
                                    color="red")
                    if float(point_r.point[2]) < 600 or float(point_r.point[2]) > -600:
                        self.points.append(point_r)
                        self.red_points.append(point_r)
                        self.red_points_dbscan.append(point_r.point[:2])
                        self.points_dbscan.append(point_r.point[:2])
                    r_cnt += 1
                cnt += 1

            self.red_size = r_cnt

        return

    def parse_raw_2d(self):
        with open(self.green_path) as g:
            g = csv.reader(g, delimiter=',')
            cnt = 0
            g_cnt = 0
            r_cnt = 0
            for row in g:
                if (cnt != 0):
                    if row[6] == '0':
                        point = Point(x=row[16], y=row[17], z='0',
                                      color="green")
                        if float(point.point[2]) < 600 or float(point.point[2]) > -600:
                            self.points.append(point)
                            self.green_points.append(point)
                            self.green_points_dbscan.append(point.point[:2])
                            self.points_dbscan.append(point.point[:2])
                            g_cnt += 1
                    else:
                        point_r = Point(x=row[16], y=row[17], z=row[18],
                                        color="red")
                        if float(point_r.point[2]) < 600 or float(point_r.point[2]) > -600:
                            self.points.append(point_r)
                            self.red_points.append(point_r)
                            self.red_points_dbscan.append(point_r.point[:2])
                            self.points_dbscan.append(point_r.point[:2])
                            r_cnt += 1
                cnt += 1

            self.green_size = g_cnt
            self.red_size = r_cnt

        return

    def points_summary(self):
        number_all = len(self.points)
        number_green = len(self.green_points)
        in_cluster_green = sum([len(g.points) for g in self.green_clusters])
        number_red = len(self.red_points)
        in_cluster_red = sum([len(r.points) for r in self.red_clusters])
        in_cluster_all = in_cluster_green + in_cluster_red
        str0 = "Done!\n\nUsing Epsilon of: " + str(self.epsilon) \
               + " and Min. neighbourhood of: " + str(self.min_n) \
               + " we were able to find:\nGreen clusters: " + str(len(self.green_clusters)) \
               + "\n" \
               + "Red clusters: " + str(len(self.red_clusters)) + "\n" + \
               "Combined clusters: " + str(len(self.clusters)) + "\n"
        str1 = "The total number of points is: " + str(number_all) + \
               " of which " + str(in_cluster_all) + " are in clusters"
        str2 = "There are " + str(number_green) + " green points" \
               + " of which " + str(float(float(in_cluster_green * 100) / float(number_green))) \
               + "% are in clusters." if number_green > 0 else "Not enough green points for this."
        str3 = "There are " + str(number_red) + " red points" \
               + " of which " + str(float(float(in_cluster_red * 100) / float(number_red))) \
               + "% are in clusters." if number_red > 0 else "Not enough red points for this."
        allstr = str0 + "\n" + str1 + "\n" + str2 + "\n" + str3 + "\n"
        self.f.write(allstr)

    # def get_centroids(self):
    #     for cluster in self.clusters:
    #         cluster.get_centroid()
    #     for red_cluster in self.red_clusters:
    #         red_cluster.get_centroid()
    #     for green_cluster in self.green_clusters:
    #         green_cluster.get_centroid()
    #     self.centers = [x.center for x in self.clusters]
    #     self.green_centers = [x.center for x in self.green_clusters]
    #     self.red_centers = [x.center for x in self.red_clusters]

    def print_f(self, string, f):
        f.write(string)


# ***-----------------------CLUSTER----------------------------------***#

class Cluster:
    def __init__(self):
        self.points = []
        self.size = 0
        self.shape = 0
        self.shape_2d = 0
        self.large_diameter = 0
        self.center = 0
        self.pca = None
        self.angle_x = 0
        self.angle_y = 0
        self.angle_z = 0
        self.is_mixed = False
        self.is_colocalized = False
        self.final_color = "no_color_yet"  # is the color of the cluster after the appendage of points from the 2nd color.

    def add_point(self, point):
        self.points.append(point)
    #
    # def get_centroid(self):
    #     x = [float(p.point[0]) for p in self.points]
    #     y = [float(p.point[1]) for p in self.points]
    #     z = [float(p.point[2]) for p in self.points]
    #     center = sum(x) / len(x), sum(y) / len(y), sum(z) / len(z)
    #     self.center = center

    # def outliers(self):  # ???? what does this do?
    #     distances = [dist(x, self.center) for x in self.points]
    #     mean_dist = sum(distances) / len(distances)
    #     std_dist = np.std(distances)

    def pca_analysis(self, dim=3,
                     sphere=True):  # if sphere is True (default) also updates the sphere score of the cluster.
        if dim > 2:
            self.pca = PCA(np.array([x.point for x in self.points], dtype=np.float))
        else:
            self.pca = PCA(np.array([x.point for x in self.points], dtype=np.float), standardize=False)

        self.center = self.pca.mu
        v1 = self.pca.Wt[0]
        self.angle_x = angle(v1, [1, 0, 0])
        self.angle_y = angle(v1, [0, 1, 0])
        self.angle_z = angle(v1, [0, 0, 1])
        min_sig = float(self.pca.sigma[0]) / self.pca.sigma[1] if self.pca.sigma[1] != 0 else np.nan
        if min_sig != np.nan and min_sig > 1:
            min_sig = 1. / min_sig
        if sphere:
            self.shape_2d = min_sig
        self.large_diameter = max(self.pca.sigma[0], self.pca.sigma[1]) * 2
        self.size = math.sqrt(self.pca.sigma[0] * self.pca.sigma[1])

    def clean_outliers(self, dim=3):
        """"returns a list of points that survived the
                filtering of 2.5 sigma."""
        ok_points = []
        not_ok_points = []

        if dim != 3:
            distances = [dist(x.point, self.center, d3=False) for x in self.points]
        else:
            distances = [dist(x.point, self.center) for x in self.points]
        stdv = np.std(distances)
        meandis = np.mean(distances)
        for i in range(len(self.points)):
            if distances[i] < (2.5 * stdv + meandis):
                ok_points.append(self.points[i])
            else:
                not_ok_points.append(self.points[i])
        return ok_points, not_ok_points
