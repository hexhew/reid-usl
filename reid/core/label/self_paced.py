from collections import defaultdict

import numpy as np
import torch
from sklearn.cluster import DBSCAN

from ..distances import jaccard_distance
from .builder import LABEL_GENERATORS


@LABEL_GENERATORS.register_module()
class SelfPacedGenerator(object):

    def __init__(self, eps, min_samples=4, use_outliers=True, k1=30, k2=6):
        assert isinstance(eps, (tuple, list))
        self.eps = sorted(list(eps))
        self.min_samples = min_samples
        self.use_outliers = use_outliers
        self.k1 = k1
        self.k2 = k2

    @torch.no_grad()
    def dbscan_single(self, features, dist, eps):
        assert isinstance(dist, np.ndarray)

        cluster = DBSCAN(
            eps=eps,
            min_samples=self.min_samples,
            metric="precomputed",
            n_jobs=-1)
        labels = cluster.fit_predict(dist)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # cluster labels -> pseudo labels
        # compute cluster centers
        centers = defaultdict(list)
        outliers = 0
        for i, label in enumerate(labels):
            if label == -1:
                if not self.use_outliers:
                    continue
                labels[i] = num_clusters + outliers
                outliers += 1

            centers[labels[i]].append(features[i])

        centers = [
            torch.stack(centers[idx], dim=0).mean(0)
            for idx in sorted(centers.keys())
        ]
        centers = torch.stack(centers, dim=0)
        labels = torch.from_numpy(labels).long()
        num_clusters += outliers

        return labels, centers, num_clusters

    @torch.no_grad()
    def dbscan_self_paced(self, features, dist, eps, indep_thres=None):
        labels_tight, _, _ = self.dbscan_single(features, dist, eps[0])
        labels_normal, _, num_classes = self.dbscan_single(
            features, dist, eps[1])
        labels_loose, _, _ = self.dbscan_single(features, dist, eps[2])

        # compute R_indep and R_comp
        N = labels_normal.size(0)
        label_sim = (
            labels_normal.expand(N, N).eq(labels_normal.expand(N,
                                                               N).t()).float())
        label_sim_tight = (
            labels_tight.expand(N, N).eq(labels_tight.expand(N,
                                                             N).t()).float())
        label_sim_loose = (
            labels_loose.expand(N, N).eq(labels_loose.expand(N,
                                                             N).t()).float())

        R_comp = 1 - torch.min(label_sim, label_sim_tight).sum(-1) / torch.max(
            label_sim, label_sim_tight).sum(-1)
        R_indep = 1 - torch.min(label_sim, label_sim_loose).sum(
            -1) / torch.max(label_sim, label_sim_loose).sum(-1)
        assert (R_comp.min() >= 0) and (R_comp.max() <= 1)
        assert (R_indep.min() >= 0) and (R_indep.max() <= 1)

        cluster_R_comp, cluster_R_indep = defaultdict(list), defaultdict(list)
        cluster_img_num = defaultdict(int)
        for comp, indep, label in zip(R_comp, R_indep, labels_normal):
            cluster_R_comp[label.item()].append(comp.item())
            cluster_R_indep[label.item()].append(indep.item())
            cluster_img_num[label.item()] += 1

        cluster_R_comp = [
            min(cluster_R_comp[i]) for i in sorted(cluster_R_comp.keys())
        ]
        cluster_R_indep = [
            min(cluster_R_indep[i]) for i in sorted(cluster_R_indep.keys())
        ]
        cluster_R_indep_noins = [
            iou for iou, num in zip(cluster_R_indep,
                                    sorted(cluster_img_num.keys()))
            if cluster_img_num[num] > 1
        ]
        if indep_thres is None:
            indep_thres = np.sort(cluster_R_indep_noins)[min(
                len(cluster_R_indep_noins) - 1,
                np.round(len(cluster_R_indep_noins) * 0.9).astype("int"),
            )]

        labels_num = defaultdict(int)
        for label in labels_normal:
            labels_num[label.item()] += 1

        centers = defaultdict(list)
        outliers = 0
        for i, label in enumerate(labels_normal):
            label = label.item()
            indep_score = cluster_R_indep[label]
            comp_score = R_comp[i]
            if label == -1:
                assert not self.use_outliers, "exists a bug"
                continue
            if (indep_score > indep_thres) or (comp_score.item() >
                                               cluster_R_comp[label]):
                if labels_num[label] > 1:
                    labels_normal[i] = num_classes + outliers
                    outliers += 1
                    labels_num[label] -= 1
                    labels_num[labels_normal[i].item()] += 1

            centers[labels_normal[i].item()].append(features[i])

        num_classes += outliers
        assert len(centers.keys()) == num_classes

        centers = [
            torch.stack(centers[idx], dim=0).mean(0)
            for idx in sorted(centers.keys())
        ]
        centers = torch.stack(centers, dim=0)

        return labels_normal, centers, num_classes, indep_thres

    @torch.no_grad()
    def gen_labels(self, features):
        dist = jaccard_distance(features, k1=self.k1, k2=self.k2)

        if len(self.eps) == 1:
            # normal clustering
            labels, centers, num_classes = self.dbscan_single(
                features, dist, self.eps[0])
            return labels, centers, num_classes, None
        elif len(self.eps) == 3:
            (labels_normal, centers, num_classes,
             indep_thres) = self.dbscan_self_paced(features, dist, self.eps)
            return labels_normal, centers, num_classes, indep_thres
