# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 00:51:18 2022

@author: zhenq
"""
import numpy as np
import pandas as pd
import igraph as ig
import leidenalg as la
from sklearn.cluster import KMeans, AgglomerativeClustering

from scgp.neighborhood import build_feature_knn_umap, construct_graph


def leiden_partition(G, rp=0.1, **kwargs):
    """Leiden partition wrapper

    Args:
        G (nx.Graph): graph of the target region(s)
        rp (float, optional): resolution parameter

    Returns:
        list: membership for the target region(s)
    """
    _G = ig.Graph.from_networkx(G)
    edge_weights = [e[2]['weight'] if 'weight' in e[2] else 1. for e in G.edges.data()]
    partition = la.find_partition(
        _G, la.CPMVertexPartition, weights=edge_weights, resolution_parameter=rp, **kwargs)
    membership = partition.membership
    return membership


def leiden_partition_with_reference(joint_G,
                                    initial_membership=None,
                                    is_membership_fixed=None,
                                    rp=2e-4,
                                    **kwargs):
    """Leiden partition on target graph with reference

    Args:
        joint_G (nx.Graph): graph of the query region + reference pseudo-nodes
        initial_membership (list/None, optional): list of initial membership
            for the target graph.
        is_membership_fixed (list/None, optional): if nodes in the graph have
            fixed membership, reference pseudo-nodes will be fixed.
        rp (float, optional): resolution parameter. Defaults to 2e-4.

    Returns:
        list: membership for the target region + reference
    """
    initial_membership = [0] * len(joint_G) if initial_membership is None else initial_membership
    is_membership_fixed = [False] * len(joint_G) if is_membership_fixed is None else is_membership_fixed

    _joint_G = ig.Graph.from_networkx(joint_G)
    edge_weights = [e[2]['weight'] if 'weight' in e[2] else 1. for e in joint_G.edges.data()]
    partition = la.CPMVertexPartition(
        _joint_G,
        initial_membership=initial_membership,
        weights=edge_weights,
        resolution_parameter=rp)

    optimiser = la.Optimiser()
    optimiser.optimise_partition(partition, is_membership_fixed=is_membership_fixed)
    membership = partition.membership
    return membership


def leiden_clustering(feature_ar, rp=0.1, k=15, seed=123, **kwargs):
    """Generic leiden partition on an array of features, used for clustering
    without spatial edges

    Args:
        feature_ar (np.array): node features
        rp (float, optional): resolution parameter.
        k (int, optional): number of nearest neighbors. Defaults to 15.
        seed (int, optional): random seed

    Returns:
        list: membership for the list of features
    """
    assert isinstance(feature_ar, np.ndarray)
    feature_df = pd.DataFrame(feature_ar)
    feature_neighbors_df = build_feature_knn_umap(feature_df, k=k, seed=seed, attach_to_object=False)

    nx_graph = construct_graph(
        feature_neighbors_df, feature_df=feature_df, weighted=True, normalize=True)

    graph = ig.Graph.from_networkx(nx_graph)
    partition_type = la.RBConfigurationVertexPartition
    partition_kwargs = {}
    partition_kwargs['n_iterations'] = -1
    partition_kwargs['seed'] = seed
    partition_kwargs['resolution_parameter'] = rp
    partition_kwargs['weights'] = np.array(graph.es['weight']).astype(np.float64)
    partition_kwargs.update(kwargs)
    part = la.find_partition(graph, partition_type, **partition_kwargs)
    return part.membership, graph


def remove_isolated_patch(membership):
    """Denoising for partition/clustering

    This method removes cluster/partition that constitutes less than 0.2 percent
    of the entire graph.

    Args:
        membership (list): cluster assignment/partition of nodes in the graph

    Returns:
        list: denoised membership list
    """
    membership = np.array(membership)
    num_nodes = len(membership)
    for k, count in zip(*np.unique(membership, return_counts=True)):
        if count <= max(1, 0.002 * num_nodes):
            membership[np.where(membership == k)] = -1
    return list(membership)


def smooth_spatially_isolated_patch(neighbor_df, membership, continuity_level=0):
    """Spatial smoothing for partition/clustering

    This method looks for cells who have different partitions from their
    neighbors and alter their partitions accordingly.

    Note that neighborhood should not contain feature edges.

    Args:
        neighbors (pd.DataFrame): (spatial) neighborhood dataframe
        membership (list): cluster assignment/partition of nodes in the graph
        continuity_level (int): threshold for smoothing, node membership will
            be altered if the number of its neighbors sharing the same membership
            is smaller than this value.

    Returns:
        list: denoised membership list
    """
    assert 'feature' not in neighbor_df
    assert 'neighbors-feature' not in neighbor_df
    membership = np.array(membership)
    all_neighbors = neighbor_df.sum(1)
    assert len(membership) == len(all_neighbors)
    membership_dict = {k: v for k, v in zip(all_neighbors.index, membership)}

    for cell_id, ns in all_neighbors.items():
        self_id = membership_dict[cell_id]
        neighbor_ids = [membership_dict[n] for n in ns if n != cell_id]
        if len(neighbor_ids) >= 2:
            if self_id not in neighbor_ids or neighbor_ids.count(self_id) <= continuity_level:
                membership_dict[cell_id] = majority_of_list([i for i in neighbor_ids if i != -1])

    membership = [membership_dict[cell_id] for cell_id in all_neighbors.index]
    return membership


def majority_of_list(ar):
    """Get the majority item (>=50%) out of an array

    Args:
        ar (list): list of items

    Returns:
        major item, -1 if there is no major item
    """
    if len(ar) == 0:
        return -1
    items_ct = dict(zip(*np.unique(ar, return_counts=True)))
    major_item = sorted(items_ct, key=lambda x: items_ct[x])[-1]
    if items_ct[major_item] >= 0.5 * len(ar):
        return major_item
    else:
        return -1


def generic_clustering(feature_df, method='KMeans', method_kwargs={'n_clusters': 6}):
    """Generic clustering on an array of features

    Args:
        feature_df (pd.DataFrame): feature dataframe
        method (str, optional): clustering method, one of: "KMeans", "AGG", "Leiden"
        method_kwargs (dict, optional): kwargs for the clustering method

    Returns:
        predicted_groups (dict): dict of cell(node) name: cluster id
        model: clustering model
    """

    if method == 'KMeans':
        assert 'n_clusters' in method_kwargs
        model = KMeans(**method_kwargs)
        membership = model.fit_predict(feature_df)

    elif method == 'AGG':
        assert 'n_clusters' in method_kwargs
        model = AgglomerativeClustering(**method_kwargs)
        membership = model.fit_predict(feature_df)

    elif method == 'Leiden':
        assert 'rp' in method_kwargs
        feature_ar = np.array(feature_df)
        original_membership, graph = leiden_clustering(feature_ar, **method_kwargs)
        membership = remove_isolated_patch(original_membership)
        model = graph

    else:
        raise ValueError("Method %s not supported" % method)

    predicted_groups = membership_to_membership_dict(membership, feature_df)
    predicted_groups = reorder_cluster_id(predicted_groups)
    return predicted_groups, model


def reorder_cluster_id(membership_dict):
    """Reorder cluster IDs based on size of clusters

    Args:
        membership_dict (dict): dict of cell(node) name: cluster id

    Returns:
        dict: reordered cluster dict
    """
    values = list(membership_dict.values())
    unique_val_ct_dict = dict(zip(*np.unique(values, return_counts=True)))
    unique_vals = sorted(unique_val_ct_dict.keys(), key=lambda x: -unique_val_ct_dict[x])

    val_mapping = {v: i for i, v in enumerate(unique_vals)}
    return {k: val_mapping[v] for k, v in membership_dict.items()}


def membership_to_membership_dict(membership, feature_df):
    """Transform list of cluster/membership to dict

    Args:
        membership (list): list of cluster/membership, in the same order as
            the index of `feature_df`
        feature_df (pd.DataFrame): feature dataframe

    Returns:
        dict: dict of cell(node) name: cluster id
    """
    assert feature_df.shape[0] == len(membership)
    membership_dict = {}
    for k, m in zip(feature_df.index, membership):
        membership_dict[k] = m
    return membership_dict


def membership_dict_to_membership(membership_dict, feature_df):
    """Transform dict of cluster/membership to list

    Args:
        membership_dict (dict): dict of cell(node) name: cluster id
        feature_df (pd.DataFrame): feature dataframe

    Returns:
        list: list of cluster/membership, in the same order as the index
            of `feature_df`
    """
    membership = []
    for k in feature_df.index:
        membership.append(membership_dict[k])
    return membership
