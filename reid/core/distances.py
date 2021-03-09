import faiss
import faiss.contrib.torch_utils
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def cosine_distance(features, others=None):
    """Computes cosine distance.

    Args:
        input1 (Tensor): 2-D feature matrix.
        input2 (Tensor): 2-D feature matrix.

    Returns:
        Tensor: distance matrix.
    """
    if others is None:
        features = F.normalize(features, p=2, dim=1)
        dist_m = 1 - torch.mm(features, features.t())
    else:
        features = F.normalize(features, p=2, dim=1)
        others = F.normalize(others, p=2, dim=1)
        dist_m = 1 - torch.mm(features, others.t())

    return dist_m


@torch.no_grad()
def euclidean_distance(features, others=None):

    if others is None:
        n = features.size(0)
        x = features.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
    else:
        m, n = features.size(0), others.size(0)
        dist_m = (
            torch.pow(features, 2).sum(dim=1, keepdim=True).expand(m, n) +
            torch.pow(others, 2).sum(dim=1, keepdim=True).expand(n, m).t())
        dist_m.addmm_(features, others.t(), beta=1, alpha=-2)

    return dist_m


@torch.no_grad()
def jaccard_distance(features, k1=20, k2=6, fp16=False):

    features = features.cuda()

    N = features.size(0)
    mat_type = np.float16 if fp16 else np.float32

    res = faiss.StandardGpuResources()
    res.setDefaultNullStreamAllDevices()
    _, initial_rank = faiss.knn_gpu(res, features, features, k1)
    initial_rank = initial_rank.cpu().numpy()

    nn_k1 = []
    nn_k1_half = []
    for i in range(N):
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(
            k_reciprocal_neigh(initial_rank, i, int(np.around(k1 / 2))))

    V = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if len(
                    np.intersect1d(candidate_k_reciprocal_index,
                                   k_reciprocal_index)
            ) > 2 / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(
            k_reciprocal_expansion_index)  # element-wise unique

        x = features[i].unsqueeze(0).contiguous()
        y = features[k_reciprocal_expansion_index]
        m, n = x.size(0), y.size(0)
        dist = (
            torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) +
            torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t())
        dist.addmm_(x, y.t(), beta=1, alpha=-2)

        if fp16:
            V[i, k_reciprocal_expansion_index] = (
                F.softmax(-dist,
                          dim=1).view(-1).cpu().numpy().astype(mat_type))
        else:
            V[i, k_reciprocal_expansion_index] = (
                F.softmax(-dist, dim=1).view(-1).cpu().numpy())

    del nn_k1, nn_k1_half, x, y
    features = features.cpu()

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=mat_type)
        for i in range(N):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe

    del initial_rank

    invIndex = []
    for i in range(N):
        invIndex.append(np.where(V[:, i] != 0)[0])  # len(invIndex)=all_num

    jaccard_dist = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        temp_min = np.zeros((1, N), dtype=mat_type)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(
                V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])

        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    del invIndex, V

    pos_bool = jaccard_dist < 0
    jaccard_dist[pos_bool] = 0.0

    return jaccard_dist


def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, :k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]

    return forward_k_neigh_index[fi]
