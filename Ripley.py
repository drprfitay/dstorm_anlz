import pandas as pd
import numpy as np
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn import metrics
import plotly
import math
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

####################################### 2D Implementation of Ripley's Functions #######################################

def Ripleys_H(pnts_df):
    """This function calculates t - L(t) and plots the results
    Args:
        pnts_df - a dataframe of points in a cluster found by DBSCAN
    """
    pnts = pnts_df[["x", "y"]]
    pnt_lst = pnts.values.tolist()
    area = Measure_area(pnt_lst)
    #print("area = ", area)
    n = len(pnt_lst)
    centroid = calc_centroid(pnt_lst, n)
    #print("centroid = ",centroid)
    max_dist = calc_max_dist(pnt_lst, centroid)
    #print("max dist = ", max_dist)
    H_vals = dict()
    L_vals = dict()
    K_vals = dict()
    rg = np.arange(1, max_dist, 30)
    for t in rg:
        filtered = t_from_centroid(pnt_lst, t, centroid)
        if len(filtered) > 2:
            K_score = Ripleys_K(filtered, t, area)
            L_score = math.sqrt(K_score / (math.pi))
            H_score = t - L_score
            #print("L_score = ", L_score)
            #print("H_score = ", H_score)
            K_vals[t] = K_score
            H_vals[t] = H_score
            L_vals[t] = L_score
    df_list = []
    for key, value in H_vals.items():
        df_list.append((key, value))
    df = pd.DataFrame(df_list, columns = ['t', 'H(t)'])
    df['L(t)'] = L_vals.values()
    df['K(t)'] = K_vals.values()
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("Ripley's K", "Ripley's L", "Ripley's H"))
    fig.add_trace(
    go.Scatter(x=df['t'], y=df['K(t)'], mode = 'lines + markers'),
    row=1, col=1
    )
    fig.add_trace(
    go.Scatter(x=df['t'], y=df['L(t)'], mode = 'lines + markers'),
    row=1, col=2
    )
    fig.add_trace(
    go.Scatter(x=df['t'], y=df['H(t)'], mode = 'lines + markers'),
    row=1, col=3
    )
    fig.update_layout(height=800, width=1400, title_text="Side By Side Subplots")
    fig.show()


def Ripleys_K(pnts, t, area):
    """ This function returns a scatter-based score within some radius
    Args:
        pnts - a list of points in a suspected cluster
        t - a radius between 0 and the maximal distance between centroid and some point in pnts
        area - the area of the convex hull containing pnts
    """
    n = len(pnts)
    #print("n = ", n)
    #print("t = ", t)
    
    lmbda = n / area
    temp_sum = 0
    i = 0
    for p1 in pnts:
        j = len(pnts)
        for p2 in pnts[::-1]:
            if j > i:
                dist = calc_ed(p1, p2)
                if dist < t: # Else indicator function is 0 and no addition is required
                    temp_sum += (1 / n)
            j -= 1
        i += 1
    K_score = temp_sum / lmbda
    return K_score


def calc_ed(pnt1, pnt2):
    """ This function calculates the Euclidean distance between 2 points
    Args:
        pnt1, pnt2 - two points with coordinates x,y
    """
    x_d = (pnt1[0] - pnt2[0])
    x_s = x_d ** 2
    y_d = (pnt1[1] - pnt2[1])
    y_s = y_d ** 2
    dist = math.sqrt(x_s + y_s)
    return dist

def calc_centroid(pnt_lst, n):
    """ This function averages all points and returns the centroid of the bunch
    Args:
        pnt_lst - a list of points in a suspected cluster
        n - number of points in pnts
    """
    x_sum = 0
    y_sum = 0
    
    for point in pnt_lst:
        x_sum += point[0]
        y_sum += point[1]

    avg_x = x_sum / n
    avg_y = y_sum / n
    centroid = [avg_x, avg_y]
    return centroid

def calc_max_dist(pnt_lst, centroid):
    """
    This function finds the maximal distance from the centroid of the cluster to an existing point in pnts
    Args:
        pnts - a list of points, a suspected "cluster"
        centroid - the central point of the "cluster"
    """
    max_dist = 0
    for point in pnt_lst:
        dist = calc_ed(centroid, point)
        if dist > max_dist:
            max_dist = dist
    return max_dist
    

def Measure_area(pnt_lst):
    """ This function returns the area of a convex hull as an approximation of the cluster's area
    Args:
        pnt_lst - a list of points in a suspected cluster
    """
    hull = []
    pnt_lst.sort(key=lambda x:[x[0],x[1]])
    start = pnt_lst.pop(0)
    hull.append(start)
    pnt_lst.sort(key=lambda p: (get_slope(p,start), -p[1],p[0]))
    for point in pnt_lst:
        hull.append(point)
        while len(hull) > 2 and get_cross_product(hull[-3],hull[-2],hull[-1]) <= 0:
            hull.pop(-2)
    m = len(hull)
    #print("Number of vertices in hull: ", m)
    #print("hull: ", hull)
    area = 0.0
    for i in range(m):
        j = (i + 1) % m
        area += (hull[i][0] * hull[j][1])
        area -= (hull[j][0] * hull[i][1])
    area1 = abs(area) / 2.0
    return area1

def get_slope(p1, p2):
    if p1[0] == p2[0]:
        return float('inf')
    else:
        return 1.0*(p1[1]-p2[1])/(p1[0]-p2[0])

def get_cross_product(p1, p2, p3):
    return ((p2[0] - p1[0])*(p3[1] - p1[1])) - ((p2[1] - p1[1])*(p3[0] - p1[0]))

def t_from_centroid(pnt_lst, t, centroid):
    """ This function filters the points list to points within distance t from the centroid
    Args:
        pnt_lst - a list of points
        t - the radius of the circle that contains all filtered points
        centroid - the central point of the "cluster"
    """
    filtered = []
    for point in pnt_lst:
        dist = calc_ed(centroid, point)
        if dist <= t:
            filtered.append(point)
    return filtered

################################################## DBSCAN Implementation ##################################################

cols_list = ["photon-count", "x", "y"]
df = pd.read_csv("CTRL abdomen WAKO 13 CS2 647.csv", usecols = cols_list)
#df = pd.read_csv("CTRL abd Wako 18 CS3 647 csv.csv", usecols = cols_list)
rslt_df = df[df["photon-count"] > 1000]
xy_pts = rslt_df[["x", "y"]]
clustering = DBSCAN(eps = 70, min_samples = 20, metric = 'euclidean', 
                  metric_params = None, algorithm = 'auto', leaf_size = 30, p = None, 
                  n_jobs = None).fit(xy_pts)
labels = clustering.labels_
xy_pts["Label"] = labels
cluster_num = len(set(labels)) - (1 if -1 in labels else 0)
noise_num = list(labels).count(-1)
unique_labels = set(labels)
c = Counter(labels)
clusters = dict()
for (key, value) in c.items():
    if key != -1:
        if value > 30:
            clusters[key] = value
sorted_values = sorted(clusters.values(), reverse = True)
sorted_clusters = dict()
for i in sorted_values:
    for k in clusters.keys():
        if clusters[k] == i:
            sorted_clusters[k] = clusters[k]
            break
non_noisy_points = xy_pts[labels != -1]
#i = 0
for (key, value) in sorted_clusters.items():
    r = non_noisy_points.loc[non_noisy_points["Label"] == key]
    print('Current Cluster: ', key)
    print(r)
    Ripleys_H(r)


########################################################### PLOT ###########################################################
"""
fig = px.scatter(non_noisy_points, x='x', y='y', color='Label', opacity = 0.3, size_max=0.1)
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))
fig.show()
"""

