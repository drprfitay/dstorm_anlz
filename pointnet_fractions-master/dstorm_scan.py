import os
import torch
import numpy as np
import pandas as pd
import threading
from sklearn.neighbors import NearestNeighbors
from IPython.display import display, HTML
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datasets.dstorm_datasets import DstormDatasetDBSCAN
import functools
from matplotlib import cm, colors
cmap = lambda v: f'rgb{tuple((np.array(colors.to_rgb(cm.YlOrRd(v)))*255).astype(np.int16))}'



filelist = ["3568.csv", "568.csv", "647.csv"]
print(f"Number of samples to choose from: 0-{len(filelist) - 1}")


indices = [0, 1, 2]
dataset = DstormDatasetDBSCAN(
    root=[filelist[i] for i in indices], 
    min_npoints=0, 
    dbscan_eps=200, 
    dbscan_min_samples=16,
    coloc_distance=50,
    coloc_neighbors=1,
    workers=2
)

orig_df = dataset.orig_df
try:
    for e in dataset.orig_df.Exception.unique():
        print(f'1) {e}')
except AttributeError as e:
    print("Done with no errors in original dataframe")

groups_df = dataset.groups_df
try:
    for e in dataset.groups_df.Exception.unique():
        print(f'1) {e}')
except AttributeError as e:
    print("Done with no errors in groups dataframe")

print(orig_df.columns)
print(orig_df.pointcloud)

colocalization_paths = orig_df.query('colocalization_available').full_path.tolist()
for ind in indices:
    if filelist[ind] in  colocalization_paths:
        print(f"{ind}:{filelist[ind]} has colocalization")

print(orig_df.loc[1].pointcloud.values)
#with pd.option_context('display.max_rows', None, 'display.max_columns', orig_df.shape[1]):
#    print(orig_df)
#with pd.option_context('display.max_rows', None, 'display.max_columns', groups_df.shape[1]):
#    print(groups_df)
#print("Hey!\n")
#print(orig_df)
#print(groups_df)
#print(orig_df[['filename', 'coprotein', 'label', 'probe0_num_of_points', 'probe0_ngroups',
#                 'probe1_num_of_points', 'probe1_ngroups']])

groups_df['pca_major_axis_std'] = groups_df['pca_std'].apply(lambda x: x[0])
groups_df['pca_minor_axis_std'] = groups_df['pca_std'].apply(lambda x: x[1])

df = groups_df.query('probe == 0').copy()
print("Probe 0 DBScan stats:")
print(df.groupby('filename').agg({
    'num_of_points': ['mean', 'median'],
    'pca_major_axis_std': ['mean', 'median'],
    'pca_minor_axis_std': ['mean', 'median'],
    'pca_size': ['mean', 'median']
}).reset_index())


df = groups_df.query('probe == 1').copy()
if len(df) > 0:
    print("Probe 1 DBScan stats:")
    print(df.groupby('filename').agg({
        'num_of_points': ['mean', 'median'],
        'pca_major_axis_std': ['mean', 'median'],
        'pca_minor_axis_std': ['mean', 'median'],
        'pca_size': ['mean', 'median']
    }).reset_index())


# group histograms
#~~~~~~~~~~~~~~~~~~~
# number of cluster points
num_of_points_dict = {
    'is_probablity': False,
    'xbins': {
        'start': None,
        'end': None,
        'size': 9000
    }
}
# stdev major axis
major_axis_dict = {
    'is_probablity': False,
    'xbins': {
        'start': None,
        'end': None,
        'size': None
    }
}
# stdev minor axis
minor_axis_dict = {
    'is_probablity': False,
    'xbins': {
        'start': None,
        'end': None,
        'size': None
    }
}
# density
density_dict = {
    'is_probablity': False,
    'xbins': {
        'start': None,
        'end': None,
        'size': None
    }
}

range_split = lambda v, s: [((v * i)/s + 1, (v * (i+1))/s) for i in range(0,s)]
count_elements_in_range = lambda r, l: len([i for i in l if i >= r[0] and i < r[1]])
freq_array = lambda p: functools.reduce(lambda a,b: a + b, [[i+1] * v for i,v in enumerate(p)])

for row, ind in enumerate(orig_df.index, 1):
    fig = make_subplots(
        rows=1, 
        cols=1, 
        vertical_spacing=0.05,
        row_titles=orig_df.filename.to_list(),
        column_titles=['Number of points in clusters', 'stdev major axis', 'stdev minor axis', 'density'],
    )
    df = groups_df.query(f"(full_path == '{orig_df.loc[ind].full_path}') and (probe == 0)")
    num_of_points = df.num_of_points.to_numpy()
    stdev = np.stack(df.pca_std.to_numpy())
    density = num_of_points/(np.pi*stdev[:,0]*stdev[:,1])
    
    rs = range_split(max(num_of_points) + 1, 15)
    counted_clusters = [count_elements_in_range(rng, num_of_points) for rng in rs]
    titles = ["%d - %d" % (rng[0], rng[1]) for rng in rs]
    print(num_of_points)
    print(counted_clusters)
    print(titles)
    fig.add_trace(go.Histogram(histfunc="sum", y=counted_clusters, x=titles, name="sum"))
    fig.update_layout(
    xaxis_title_text='Range of points found', # xaxis label
    yaxis_title_text='Clusters count', # yaxis label
    bargap=0)
    #fig.add_trace(go.Histogram(
    #    x=stdev[:,0], 
    #    histnorm='probability' if major_axis_dict['is_probablity'] else None, 
    #    xbins=major_axis_dict['xbins']
    #), row=1, col=2)
    #fig.add_trace(go.Histogram(
    #    x=stdev[:,1], 
    #    histnorm='probability' if minor_axis_dict['is_probablity'] else None, 
    #    xbins=minor_axis_dict['xbins']
    #), row=1, col=3)
    #fig.add_trace(go.Histogram(
    #    x=density, 
    #    histnorm='probability' if density_dict['is_probablity'] else None, 
    #    xbins=density_dict['xbins']
    #), row=1, col=4)
    #fig.update_layout( 
    #    height=400*len(indices), 
    #    width=2000,
    #    showlegend=False
    #)
    fig.show()
    
    

for text in fig['layout']['annotations']:
    if text['text'][-4:] == '.csv':
        text['font']['size'] = 12
#

# k-distance histogram (0-32)
k = 16
is_3D = True

fig = make_subplots(
    rows=len(indices), 
    cols=1, 
    vertical_spacing=0.05,
    subplot_titles=orig_df.filename.to_list(),
    x_title='Number of Localizations',
    y_title='Distance (nm)'
)

for row, ind in enumerate(orig_df.index, 1):
    pc = orig_df.loc[ind].pointcloud.query("probe == 0")
    
    if is_3D:
        points = pc[['x', 'y' ,'z']].to_numpy()
    else:
        points = pc[['x', 'y']].to_numpy()
    
    nbrs = NearestNeighbors(n_neighbors=33, algorithm='ball_tree').fit(points)
    distances, _ = nbrs.kneighbors(points)

    t = [p[k] for p in distances]
    #print(distances[0:4])
    #print(t[0:4])
    t.sort()
    
    fig.add_trace(
        go.Scattergl(
            x=np.arange(len(t)),
            y=t,
            mode='lines+markers',
            marker=dict(
                color='red',           # set color to an array/list of desired values
                opacity=1
            )
        ),
        row=row, col=1
    )

fig.update_layout( 
    height=600*len(indices), 
    showlegend=False
)

fig = make_subplots(
    rows=len(indices), 
    cols=1, 
    vertical_spacing=0.05,
    subplot_titles=orig_df.filename.to_list()
)

for row, ind in enumerate(orig_df.index, 1):
    # Probe 0
    fig.add_trace(
        go.Scattergl(
            x=orig_df.loc[ind].pointcloud.query('probe == 0')['x'].values,
            y=orig_df.loc[ind].pointcloud.query('probe == 0')['y'].values,
            mode='markers',
            marker=dict(
                color='red',           # set color to an array/list of desired values
                opacity=1
            )
        ),
        row=row, col=1
    )
    
    # Probe 1
    fig.add_trace(
        go.Scattergl(
            x=orig_df.loc[ind].pointcloud.query('probe == 1')['x'].values,
            y=orig_df.loc[ind].pointcloud.query('probe == 1')['y'].values,
            mode='markers',
            marker=dict(
                color='limegreen',           # set color to an array/list of desired values
                opacity=1
            )
        ),   
        row=row, col=1
    )
    
    fig.update_xaxes(range=[0, 18e3], row=row, col=1)     
    fig.update_yaxes(scaleanchor = "x" if row == 1 else f"x{row}", scaleratio = 1, row=row, col=1)
    
fig.update_layout( 
    height=600*len(indices), 
    showlegend=False
)

fig = make_subplots(
    rows=len(indices), 
    cols=1, 
    vertical_spacing=0.05,
    subplot_titles=orig_df.filename.to_list()
)

for row, ind in enumerate(orig_df.index, 1):
    df = groups_df.query(f"(full_path == '{orig_df.loc[ind].full_path}') and (probe == 0)")
    
    # Probe 0
    fig.add_trace(
        go.Scattergl(
            x=orig_df.loc[ind].pointcloud.query('probe == 0')['x'].values,
            y=orig_df.loc[ind].pointcloud.query('probe == 0')['y'].values,
            mode='markers',
            marker=dict(
                color='grey',           # set color to an array/list of desired values
                opacity=0.1
            )
        ),
        row=row, col=1
    )
    
    # Draw clusters
    for i in df.index:
        pc = df.loc[i].pointcloud
        
        fig.add_trace(
            go.Scattergl(
                x=pc['x'].values,
                y=pc['y'].values,
                mode='markers',
                marker=dict(
                    color=i, #cmap(np.sqrt(colocalization[i])),    
                    colorscale='rainbow',
                    opacity=0.5
                )
            ),
            row=row, col=1
        )
    
    fig.update_xaxes(range=[0, 18e3], row=row, col=1)     
    fig.update_yaxes(scaleanchor = "x" if row == 1 else f"x{row}", scaleratio = 1, row=row, col=1)
    
fig.update_layout( 
    height=600*len(indices), 
    showlegend=False
)



# fig = make_subplots(
#     rows=len(indices), 
#     cols=1, 
#     vertical_spacing=0.05,
#     subplot_titles=orig_df.filename.to_list()
# )

# for row, ind in enumerate(orig_df.index, 1):
#     colocalization = orig_df.loc[ind].pointcloud.colocalization.to_numpy()
#     colocalization = colocalization / colocalization.max()
    
#     # Probe 1
#     fig.add_trace(
#         go.Scattergl(
#             x=orig_df.loc[ind].pointcloud.query('probe == 1')['x'].values,
#             y=orig_df.loc[ind].pointcloud.query('probe == 1')['y'].values,
#             mode='markers',
#             marker=dict(
#                 color='grey',           # set color to an array/list of desired values
#                 opacity=0.5
#             )
#         ),
#         row=row, col=1
#     )
    
#     # Draw clusters
# #     for group in orig_df.loc[ind].probe0_groups_df.group:
#     pc = orig_df.loc[ind].pointcloud.query("probe == 0")
        
#     fig.add_trace(
#         go.Scattergl(
#             x=pc['x'].values,
#             y=pc['y'].values,
#             mode='markers',
#             marker=dict(
#                 color=pc.colocalization.values,
#                 colorscale='ylorrd',
#                 opacity=0.5
#             )
#         ),
#         row=row, col=1
#     )
    
    
#     fig.update_xaxes(range=[0, 18e3], row=row, col=1)     
#     fig.update_yaxes(scaleanchor = "x" if row == 1 else f"x{row}", scaleratio = 1, row=row, col=1)
    
# fig.update_layout( 
#     height=600*len(indices), 
#     showlegend=False
# )




            