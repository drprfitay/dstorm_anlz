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


################################################# Ripley Optimization Functions ################################################

def is_concave(H_vals_lst):
   """
   This function finds all local maximum points (t, H(t)) and counts them.
   Args:
      H_vals_lst - a list of points which were identified as a cluster by the clustering algorithm.
   Ret:
      max_pts - a list of all maximum points found in H_vals_lst
      boo - True if there is only one maximum point, indicating that the graph is concave.
            False if there isn't a maximum point or there is more than one - indicating that the graph isn't concave.
   """
   sorted_list = sorted(H_vals_lst, key = lambda x: x[0])
   a = sorted_list[0]
   b = sorted_list[1]
   c = sorted_list[2]
   ctr = 0
   max_pts = []
   boo = True
   for i in range(3, len(sorted_list)):
      if(a[1] < b[1]) & (b[1] > c[1]):    # We want to find a better classification
         ctr += 1
         max_pts.append(b)
      a = b
      b = c
      c = sorted_list[i]
   if(ctr != 1):
      boo = False
   return max_pts, boo


####################################### 2D Implementation of Ripley's Functions #######################################
def Ripleys_H(pnts_df):
   """This function calculates t - L(t) and plots the results
   Args:
   pnts_df - a dataframe of points in a cluster found by DBSCAN
   """
   pnts = pnts_df[["x", "y"]]
   pnt_lst = pnts.values.tolist()
   area = Measure_area(pnt_lst)
   n = len(pnt_lst)
   centroid = calc_centroid(pnt_lst, n)
   max_dist = calc_max_dist(pnt_lst, centroid)
   H_vals = dict()
   L_vals = dict()
   K_vals = dict()
   step = 0.08 * max_dist
   rg = np.arange(0, max_dist / 2, step)
   #print(rg)
   for t in rg:
      filtered = t_from_centroid(pnt_lst, t, centroid)
      if len(filtered) >= 0:
         K_score = Ripleys_K(filtered, t, area, n)
         L_score = math.sqrt(K_score / (math.pi))
         H_score = L_score - t
         #print("L_score = ", L_score)
         #print("H_score = ", H_score)
         K_vals[t] = K_score
         H_vals[t] = H_score
         L_vals[t] = L_score
   df_list = []
   df_Klist = []
   df_Llist = []
   for key, value in H_vals.items():
      df_list.append((key, value))
   for key, value in K_vals.items():
      df_Klist.append((key, value))
   for key, value in L_vals.items():
      df_Llist.append((key, value))
   max_pt = None
   boo = False
   if len(df_list) > 2:                   # Otherwise the function cannot be concave
      max_lst, boo = is_concave(df_list)
      #print("is concave: ", boo)
      #print("max points = ", max_lst)
      if len(max_lst) >= 1:
         max_pt = max(max_lst, key = lambda x: x[1])
         #print("Maximal Points = ", max_pt)
   df = pd.DataFrame(df_list, columns = ['t', 'H(t)'])
   dfK = pd.DataFrame(df_Klist, columns = ['t', 'K(t)'])
   dfL = pd.DataFrame(df_Llist, columns = ['t', 'L(t)'])
   #df['L(t)'] = L_vals.values()
   #df['K(t)'] = K_vals.values()
   #plot_K(df)
   """fig = make_subplots(rows=2, cols=3,
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

   fig.add_trace(
      go.Scatter(x=pnts_df[["x", "y"]].to_numpy()[:,0].tolist(), y=pnts_df[["x", "y"]].to_numpy()[:,1].tolist(), mode = 'markers'),
      row=2, col=1)

   fig.update_layout(height=800, width=1400, title_text="Side By Side Subplots")
   fig.show() """
   return max_pt, boo, df, dfK, dfL


def Ripleys_K(pnts, t, area, total_pts):
    """ This function returns a scatter-based score within some radius
    Args:
        pnts - a list of points in a suspected cluster
        t - a radius between 0 and the maximal distance between centroid and some point in pnts
        area - the area of the convex hull containing pnts
    """
    n = len(pnts)
    lmbda = total_pts / area
    temp_sum = 0
    i = 0
    for p1 in pnts:
        j = len(pnts)
        for p2 in pnts[::-1]:
            if j > i:
                dist = calc_ed(p1, p2)
                if dist < t: # Else indicator function is 0 and no addition is required
                    temp_sum += (1 / n) # This gives a very good estimation for weight factor
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

########################################################### PLOT ###########################################################


def plot_all_points(xy_pts, file_name):
   fig2 = px.scatter(xy_pts, x='x', y='y', opacity = 0.3, size_max=0.1, title = file_name)
   fig2.update_layout(legend=dict(
       yanchor="top",
       y=0.99,
       xanchor="left",
       x=0.01
   ))
   fig2.show()

def plot_DBSCAN_res(non_noisy_points, file_name, radius, i):
   if i == 0:
      ttl = file_name + " With Radius " + str(radius)
   if i == 1:
      ttl = file_name + " With Radius " + str(radius)
   if i == 2:
      ttl = file_name + " With Radius " + str(radius)
   fig = px.scatter(non_noisy_points, x='x', y='y', color='Label', opacity = 0.3, size_max=0.1,
                    title = ttl)
   fig.update_layout(legend=dict(
       yanchor="top",
       y=0.99,
       xanchor="left",
       x=0.01
   ))
   fig.show()

def plot_Ks(K_df, cluster_ctr):
   clstrs = []
   for i in range(cluster_ctr):
      k = K_df.loc[K_df["Cluster"] == i]
      if k.empty:
         continue
      else:
         clstrs.append(k)
   nm = len(clstrs)
   cls = math.ceil(nm / 3)
   fig0 = make_subplots(rows = 3, cols = cls)
   for i in range(nm):
      r = math.ceil((i+1)/cls)
      print("r = ", r)
      c = (i % cls) + 1
      print("c = ", c)
      k1 = clstrs[i]
      fig0.add_trace(
         go.Scatter(x = k1['t'], y = k1['K(t)'], mode = 'lines + markers'),
         row = r, col = c
         )
   fig0.update_layout(height = 1100, width = 1600, title_text = "Ripley's K plots")
   fig0.show()

def plot_Ls(L_df, cluster_ctr):
   clstrs = []
   for i in range(cluster_ctr):
      l = L_df.loc[L_df["Cluster"] == i]
      if l.empty:
         continue
      else:
         clstrs.append(l)
   nm = len(clstrs)
   cls = math.ceil(nm / 3)
   fig0 = make_subplots(rows = 3, cols = cls)
   for i in range(nm):
      r = math.ceil((i+1)/cls)
      print("r = ", r)
      c = (i % cls) + 1
      print("c = ", c)
      l1 = clstrs[i]
      fig0.add_trace(
         go.Scatter(x = l1['t'], y = l1['L(t)'], mode = 'lines + markers'),
         row = r, col = c
         )
   fig0.update_layout(height = 1100, width = 1600, title_text = "Ripley's L plots")
   fig0.show()

def plot_Hs(H_df, cluster_ctr):
   clstrs = []
   for i in range(cluster_ctr):
      h = H_df.loc[H_df["Cluster"] == i]
      if h.empty:
         continue
      else:
         clstrs.append(h)
   nm = len(clstrs)
   cls = math.ceil(nm / 3)
   fig = make_subplots(rows = 3, cols = cls)
   for i in range(nm):
      r = math.ceil((i+1)/cls)
      print("r = ", r)
      c = (i % cls) + 1
      print("c = ", c)
      h1 = clstrs[i]
      fig.add_trace(
         go.Scatter(x = h1['t'], y = h1['H(t)'], mode = 'lines + markers'),
         row = r, col = c
         )
   fig.update_layout(height = 1100, width = 1600, title_text = "Ripley's H plots")
   fig.show()

################################################## DBSCAN Implementation ##################################################

def run_DBSCAN(df, file_name):
   
   rslt_df = df[df["photon-count"] > 1000]
   xy_pts = rslt_df[["x", "y"]]
   plot_all_points(xy_pts, file_name)
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
   max_pts = []
   cluster_ctr = 0
   for (key, value) in sorted_clusters.items():
       r = non_noisy_points.loc[non_noisy_points["Label"] == key]
       print('Current Cluster: ', key)
       print(r)
       cluster_ctr += 1
       max_pt, boo, temp_Hdf, temp_Kdf, temp_Ldf = Ripleys_H(r)
       #print("Max point: ", max_pt)
       if boo == True:
          max_pts.append(max_pt)
       if cluster_ctr == 1:
          H_df = temp_Hdf
          H_df['Cluster'] = key

          K_df = temp_Kdf
          K_df['Cluster'] = key

          L_df = temp_Ldf
          L_df['Cluster'] = key
       else:
          temp_Hdf['Cluster'] = key
          Hframes = [temp_Hdf, H_df]
          H_df = pd.concat(Hframes)

          temp_Kdf['Cluster'] = key
          Kframes = [temp_Kdf, K_df]
          K_df = pd.concat(Kframes)

          temp_Ldf['Cluster'] = key
          Lframes = [temp_Ldf, L_df]
          L_df = pd.concat(Lframes)
   avg = 0
   n = len(max_pts)
   for i in range(len(max_pts)):
      if(max_pts[i] != None):
         avg += max_pts[i][0]
      else:
         n -= 1
   if n != 0:
      avg = avg / n
   # From https://core.ac.uk/download/pdf/82273619.pdf article: the radius of maximal aggregation varies between R and 2R
   epsi = 9 * avg / 5
   print("max points list = ", max_pts)
   print("Epsilon <= ", epsi, "Epsilon >= ", avg)
   clustering1 = DBSCAN(eps = avg, min_samples = 20, metric = 'euclidean', 
                     metric_params = None, algorithm = 'auto', leaf_size = 30, p = None, 
                     n_jobs = None).fit(xy_pts)
   labels1 = clustering1.labels_
   xy_pts["Label"] = labels1
   cluster_num = len(set(labels1)) - (1 if -1 in labels1 else 0)
   noise_num1 = list(labels1).count(-1)
   unique_labels1 = set(labels1)
   c = Counter(labels1)
   clusters1 = dict()
   for (key, value) in c.items():
       if key != -1:
           if value > 30:
               clusters1[key] = value
   sorted_values1 = sorted(clusters1.values(), reverse = True)
   sorted_clusters1 = dict()
   for i in sorted_values1:
       for k in clusters1.keys():
           if clusters1[k] == i:
               sorted_clusters1[k] = clusters1[k]
               break
   non_noisy_points1 = xy_pts[labels1 != -1]
   for (key, value) in sorted_clusters1.items():
       r = non_noisy_points1.loc[non_noisy_points1["Label"] == key]
       #print('Current Cluster: ', key)
       #print(r)

   clustering2 = DBSCAN(eps = epsi, min_samples = 20, metric = 'euclidean', 
                     metric_params = None, algorithm = 'auto', leaf_size = 30, p = None, 
                     n_jobs = None).fit(xy_pts)
   labels1 = clustering2.labels_
   xy_pts["Label"] = labels1
   cluster_num = len(set(labels1)) - (1 if -1 in labels1 else 0)
   noise_num1 = list(labels1).count(-1)
   unique_labels1 = set(labels1)
   c = Counter(labels1)
   clusters1 = dict()
   for (key, value) in c.items():
       if key != -1:
           if value > 30:
               clusters1[key] = value
   sorted_values1 = sorted(clusters1.values(), reverse = True)
   sorted_clusters1 = dict()
   for i in sorted_values1:
       for k in clusters1.keys():
           if clusters1[k] == i:
               sorted_clusters1[k] = clusters1[k]
               break
   non_noisy_points2 = xy_pts[labels1 != -1]
   for (key, value) in sorted_clusters1.items():
       r = non_noisy_points2.loc[non_noisy_points2["Label"] == key]

   #plot_Ks(K_df, cluster_ctr)
   #plot_Ls(L_df, cluster_ctr)
   plot_Hs(H_df, cluster_ctr)
   plot_DBSCAN_res(non_noisy_points, file_name, 70, 0)
   plot_DBSCAN_res(non_noisy_points1, file_name, avg, 1)
   plot_DBSCAN_res(non_noisy_points2, file_name, epsi, 2)


cols_list = ["photon-count", "x", "y"]
#file_name = "./experimentfiles/CTRL_abdomen_WAKO/CTRL abdomen WAKO 13 CS2 647"
#file_name = "C002 WAKO 10 cs2 647 CONF 0.9 denoise 70"
#file_name = "C002 WAKO 12 cs2 647 CONF 0.9 denoise 70"
#file_name = "C004 WAKO 3 cs1 647 CONF 0.9 denoise 70"
#file_name = "C004 WAKO 6 cs1 647 CONF 0.9 denoise 70"
#file_name = "C004 WAKO 8 cs1 647 CONF 0.9 denoise 70"
file_name = "HW2 WAKO 24 cs2 647 CONF 0.9 denoise 70"
#file_name = "NN3 WAKO 2 cs2 647 CONF 0.9 denoise 70"
#file_name = "NN3 WAKO 6 cs2 647 CONF 0.9 denoise 70"
#file_name = "CTRL abd Wako 18 CS3 647 csv"
#file_name = "CTRL abdomen WAKO 13 CS2 647"
df = pd.read_csv(file_name + ".csv", usecols = cols_list) # Good results with 9/5 radius

run_DBSCAN(df, file_name)

