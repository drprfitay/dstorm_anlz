import torch


def xyz_center(xyz):
    """
    Input:
        xyz: source points, [N, 3]
    Output:
        xyz: per-point square distance, [N, 3]
        radius: max distance before normalizing
        centeroid: center of points
    """
    centroid = torch.mean(xyz, dim=0)
    xyz = xyz - centroid
    return xyz, centroid


def xyz_normalize(xyz):
    """
    Input:
        xyz: source points, [N, 3]
    Output:
        xyz: per-point square distance, [N, 3]
        radius: max distance before normalizing
        centeroid: center of points
    """
    # centroid = np.mean(xyz, axis=0)
    # xyz = xyz - centroid
    # radius = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
    # xyz = xyz / radius

    centroid = torch.mean(xyz, dim=0)
    xyz = xyz - centroid
    radius = torch.max(torch.sqrt(torch.sum(xyz ** 2, dim=1)))
    xyz = xyz / radius

    return xyz, radius, centroid


def square_distance(src, dst):
    """
    B - Batch, N,M - number of points, d-dim coordinates
    Input:
        src: source points, [B, N, d]
        dst: target points, [B, M, d]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def farthest_point_sample(xyz, npoint):
    """
    B - Batch, N, npoint - number of points, d-dim coordinates
    Input:
        xyz: pointcloud data, [B, N, d]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud data, [B, npoint, d]
    """
    device = xyz.device
    B, N, d = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance_mask = torch.ones(B, N).type_as(xyz).to(device) * 1e10

    nan_mask = (xyz!=xyz)[...,0]
    distance_mask[nan_mask] = 0
    minimum_not_nan_index = (~nan_mask).sum(dim=1).min(dim=0)[0]-1

    farthest = torch.randint(0, minimum_not_nan_index, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid)**2, -1)
        mask = (dist < distance_mask)
        distance_mask[mask] = dist[mask]
        farthest = torch.max(distance_mask, -1)[1]
    return centroids


def index_points(points, idx):
    """
    B - Batch, N,n - number of points, d-dim coordinates,
    Input:
        points: input points data, [B, N, d]
        idx: sample index data, [B, D1, D2, ..., Dn] - D1,D2,..Dn indices of points
    Return:
        new_points:, indexed points data, [B, n, d]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    B - Batch, N,M,nsample - number of points, d-dim coordinates
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, d]
        new_xyz: query points, [B, M, d]
    Return:
        group_idx: grouped points index, [B, M, nsample]
    """
    device = xyz.device
    B, N, d = xyz.shape
    _, M, _ = new_xyz.shape

    # Sampling by nearest neighbors to center
    sqrdists = square_distance(new_xyz, xyz)
    sqrdists, group_idx = torch.sort(sqrdists)

    sqrdists = sqrdists[:, :, :nsample]
    group_idx = group_idx[:, :, :nsample]

    # In case there are less than nsample points inside the ball, replace all exceeds indices with the first one
    # First point will appear multiple times.
    group_idx[sqrdists > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0]
    group_first = group_idx[:, :, 0].view(B, M, 1).repeat([1, 1, nsample])
    mask = (group_idx == N)
    group_idx[mask] = group_first[mask]

    # # Sparse sampling inside the ball
    # group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, M, 1])
    # sqrdists = square_distance(new_xyz, xyz)
    # group_idx[sqrdists > radius**2] = N
    # group_idx = group_idx.sort(dim=-1)[0][:,:,:nsample]
    #
    # # In case there are less than nsample indices in the ball, replace all exceeds indices with the first one
    # # First point will appear multiple times.
    # group_first = group_idx[:,:,0].view(B, M, 1).repeat([1, 1, nsample])
    # mask = group_idx == N
    # group_idx[mask] = group_first[mask]

    return group_idx


def sample_and_group(nsample, radius, npoint, xyz, points, rsample=1):
    """
    B - Batch, N,M,nsample - number of points, d-dim coordinates, C-dim point feature
    Input:
        nsample: number of sample
        radius: local region radius
        npoint: max number of points in local region
        xyz: input points position data, [B, N, d]
        points: input points data, [B, N, C]
        rsample: max radius to sample center points from
    Return:
        new_xyz: sampled points position data, [B, nsample, d]
        new_points: sampled points data, [B, nsample, npoint, d+C]
    """
    B, N, d = xyz.shape

    if rsample < 1:
        xyz_radius = torch.sum(xyz ** 2, -1).view(B,N,1).repeat([1,1,d])
        xyz_inner = xyz.clone()
        xyz_inner[xyz_radius > rsample**2] = 0
        new_xyz = index_points(xyz_inner, farthest_point_sample(xyz_inner, nsample))
    else:
        new_xyz = index_points(xyz, farthest_point_sample(xyz, nsample))

    idx = query_ball_point(radius, npoint, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz -= new_xyz.view(B, nsample, 1, d)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, d]
        points: input points data, [B, N, C]
    Return:
        new_xyz: sampled points position data, [B, 1, d]
        new_points: sampled points data, [B, 1, N, d+C]
    """
    device = xyz.device
    B, N, d = xyz.shape

    new_xyz = torch.zeros(B, 1, d).to(device)
    grouped_xyz = xyz.view(B, 1, N, d)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points