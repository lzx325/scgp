# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 20:48:55 2022

@author: zhenq
"""
import numpy as np
import pandas as pd
import time
from collections import Counter

from scgp.object_io import (
    get_name,
    get_cell_ids,
    get_biomarkers,
    has_feature,
    get_feature,
    extract_feature_neighborhood_with_region_cell_ids,
    assign_annotation_dict_to_objects,
)
from scgp.neighborhood import (
    build_feature_knn_umap,
    construct_graph,
    count_edges_in_neighborhood,
    build_delaunay_triangulation_neighborhood,
)
from scgp.features import calculate_feature, reweigh_features
from scgp.partition import (
    leiden_partition_with_reference,
    remove_isolated_patch,
    smooth_spatially_isolated_patch,
    membership_to_membership_dict,
)


def select_pseudo_nodes(objs,
                        membership_dict,
                        exclude_features=[],
                        feature_weights={},
                        use_partitions=None,
                        k=200,
                        intra_knn=40):
    """Select pseudo nodes from each reference partition.

    Nodes (from the reference samples) whose features
    are close to the group median of that partition are picked.

    Args:
        objs (AnnData/EMObject/list): (list of) region object(s)
        membership_dict (dict): dict of cell(node) name: cluster id
        exclude_features (list, optional): list of features to exclude.
        feature_weights (dict, optional): dict of feature weights.
            See `features.reweigh_features` for details.
        use_partitions (list/set/None, optional): list of partition ids for
            generating the pseudo nodes. All partitions will be considered
            if `None` is provided.
        k (int/float, optional): number of reference nodes per partition.
            if k >= 1, will have int(k) nodes per partition;
            if k < 1, will have int(k*size(partition)) nodes per partition.
        intra_knn (int, optional): number of feature knn edges for each
            partition in the reference graph.

    Returns:
        tuple: a tuple of three pd.DataFrame - feature dataframe, membership
            dataframe and neighborhood dataframe of the pseudo nodes.
    """
    objs = [objs] if not isinstance(objs, list) else objs

    if use_partitions is None:
        use_partitions = set(list(membership_dict.values())) - set([-1])  # Excluding outlying class
    else:
        use_partitions = set(use_partitions)
    feats_by_partition = {v: [] for v in use_partitions}
    nodes_by_partition = {v: [] for v in use_partitions}

    # Sort feature vectors by partition
    for obj in objs:
        assert has_feature(obj), "Calculate features (expression) first"
        region_id = get_name(obj)
        cell_ids = get_cell_ids(obj)
        feature_df = reweigh_features(
            get_feature(obj),
            exclude_features=exclude_features,
            feature_weights=feature_weights)

        for cell_id in cell_ids:
            p = membership_dict[(region_id, cell_id)]
            feat = np.array(feature_df.loc[cell_id])
            if p in use_partitions:
                feats_by_partition[p].append(feat)
                nodes_by_partition[p].append(('PN-' + region_id, 'PN-' + cell_id))

    feature_columns = get_feature(objs[0]).columns
    ref_feat_df = []
    ref_membership_df = []
    for p in use_partitions:
        # Find median feature vector for this partition
        feats = np.stack(feats_by_partition[p], 0)
        med_f = np.median(feats, axis=0, keepdims=True)
        _k = k if k >= 1 else len(nodes_by_partition[p]) * k
        # Select representative nodes based on their Euclidean distances to `med_f`
        ref_nodes_dist = np.linalg.norm(feats - med_f, ord=2, axis=1)

        ref_nodes_is = np.argsort(ref_nodes_dist)[:int(_k)]

        ref_nodes = np.array(nodes_by_partition[p])[ref_nodes_is]
        ref_nodes_feat = np.array(feats_by_partition[p])[ref_nodes_is]
        ref_nodes_df = pd.DataFrame(ref_nodes_feat, columns=feature_columns)
        ref_nodes_df['region_id'] = ref_nodes[:, 0]
        ref_nodes_df['CELL_ID'] = ref_nodes[:, 1]
        ref_nodes_df.set_index(['region_id', 'CELL_ID'], inplace=True)
        ref_feat_df.append(ref_nodes_df)
        ref_nodes_partition = pd.DataFrame([[p]] * len(ref_nodes_is), index=ref_nodes_df.index)
        ref_membership_df.append(ref_nodes_partition)
    ref_feat_df = pd.concat(ref_feat_df, axis=0)
    ref_membership_df = pd.concat(ref_membership_df, axis=0)

    # Add feature knn edges
    if intra_knn > 0:
        ref_neighborhood_df = build_feature_knn_umap(ref_feat_df, k=int(intra_knn), attach_to_object=False)
    return (ref_feat_df, ref_membership_df, ref_neighborhood_df)


def make_pseudo_nodes(objs,
                      membership_dict,
                      exclude_features=[],
                      feature_weights={},
                      use_partitions=None,
                      k=200,
                      intra_knn=40):
    """Generate pseudo-nodes for each partition.

    Pseudo-nodes are generated by random sampling the
    multivariate normal distribution defined by the mean and covariance
    of node features in each partition.

    Args:
        objs (AnnData/EMObject/list): (list of) region object(s)
        membership_dict (dict): dict of cell(node) name: cluster id
        exclude_features (list, optional): list of features to exclude.
        feature_weights (dict, optional): dict of feature weights.
            See `features.reweigh_features` for details.
        use_partitions (list/set/None, optional): list of partition ids for
            generating the reference graph. All partitions will be considered
            if `None` is provided.
        k (int/float, optional): number of reference nodes per partition.
            if k >= 1, will have int(k) nodes per partition;
            if k < 1, will have int(k*size(partition)) nodes per partition.
        intra_knn (int, optional): number of feature knn edges for each
            partition in the reference graph.

    Returns:
        tuple: a tuple of three pd.DataFrame - feature dataframe, membership
            dataframe and neighborhood dataframe of the reference nodes/cells.
    """
    objs = [objs] if not isinstance(objs, list) else objs

    if use_partitions is None:
        use_partitions = set(list(membership_dict.values())) - set([-1])  # Excluding outlying class
    else:
        use_partitions = set(use_partitions)
    feats_by_partition = {v: [] for v in use_partitions}
    nodes_by_partition = {v: [] for v in use_partitions}

    # Sort feature vectors by partition
    for obj in objs:
        assert has_feature(obj), "Calculate features (expression) first"
        region_id = get_name(obj)
        cell_ids = get_cell_ids(obj)
        feature_df = reweigh_features(
            get_feature(obj),
            exclude_features=exclude_features,
            feature_weights=feature_weights)

        for cell_id in cell_ids:
            p = membership_dict[(region_id, cell_id)]
            feat = np.array(feature_df.loc[cell_id])
            if p in use_partitions:
                feats_by_partition[p].append(feat)
                nodes_by_partition[p].append((region_id, cell_id))

    feature_columns = get_feature(objs[0]).columns
    pn_ct_cur = 0
    pn_feat_df = []
    pn_membership_df = []
    for p in use_partitions:
        feats = np.stack(feats_by_partition[p], 0)
        mean_f = np.mean(feats, axis=0)
        cov_f = np.cov(feats, rowvar=False)
        pn_ct = int(k) if k >= 1 else int(len(nodes_by_partition[p]) * k)
        if pn_ct == 1:
            pseudo_node_feat = mean_f.reshape((1, -1))
        else:
            pseudo_node_feat = np.random.multivariate_normal(mean_f, cov_f, size=pn_ct)

        pseudo_node_feat = pd.DataFrame(pseudo_node_feat, columns=feature_columns)
        pseudo_node_feat['region_id'] = ['PN-partition-%s' % str(p)] * len(pseudo_node_feat)
        pseudo_node_feat['CELL_ID'] = ['PN-node-%d' % cid for cid in np.arange(pn_ct_cur, pn_ct_cur + pn_ct)]
        pn_ct_cur += pn_ct
        pseudo_node_feat.set_index(['region_id', 'CELL_ID'], inplace=True)
        pn_feat_df.append(pseudo_node_feat)
        pn_membership_df.append(pd.DataFrame([[p]] * pn_ct, index=pseudo_node_feat.index))
    pn_feat_df = pd.concat(pn_feat_df, axis=0)
    pn_membership_df = pd.concat(pn_membership_df, axis=0)

    if intra_knn > 0:
        pn_neighborhood_df = build_feature_knn_umap(pn_feat_df, k=int(intra_knn), attach_to_object=False)
    return (pn_feat_df, pn_membership_df, pn_neighborhood_df)


def map_to_pseudo_nodes(query_feat_df, ref_dfs, knn=5):
    """Assign membership to nodes in query dataframe based on the partitions of
    their k-nearest neighbors in the query nodes

    Args:
        query_feat_df (pd.DataFrame): feature dataframe for query cells/nodes
        ref_dfs (tuple): a tuple of three pseudo nodes pd.DataFrame, see
            outputs of functions `select_pseudo_nodes` and `make_pseudo_nodes`.
        knn (int, optional): number of nearest neighbors. Defaults to 5.

    Returns:
        list: membership for the query cells/nodes
    """
    ref_feat_df, ref_membership_df, _ = ref_dfs
    query_membership = []
    ref_feat_ar = np.array(ref_feat_df)
    ref_membership_ar = np.array(ref_membership_df)
    for _, feat in query_feat_df.iterrows():
        dist = np.linalg.norm(np.array(feat).reshape((1, -1)) - ref_feat_ar, ord=2, axis=1)
        nn_assigns = ref_membership_ar[np.argsort(dist)[:knn]]
        query_membership.append(Counter(list(nn_assigns.flatten())).most_common()[0][0])
    return query_membership


def build_sample_reference_edges(query_feat_df, ref_feat_df, inter_knn=3, ratio=0.5):
    """Extract reference-query nearest neighbor feature edges

    Args:
        query_feat_df (pd.DataFrame): feature dataframe for query cells/nodes
        ref_feat_df (pd.DataFrame): feature dataframe for pseudo nodes
        inter_knn (int, optional): number of nearest neighbors to consider
            for reference-query feature edges
        ratio (float, optional): subsample ratio of reference-query feature edges,
            these edges will be ranked based on similarity and the `ratio` most
            similar edges will be kept. Should be between 0 and 1.

    Returns:
        pd.DataFrame: reference-query nearest neighbors
    """
    inter_knn = int(inter_knn)

    query_reference_neighbor_df = {cid: [] for cid in query_feat_df.index}
    query_reference_edges = {}
    dist_to_ref = []
    for cid, feat in query_feat_df.iterrows():
        dist = np.linalg.norm(np.array(feat) - np.array(ref_feat_df), ord=2, axis=1)
        dist_to_ref.append(np.sort(dist)[:inter_knn].mean())
        for i in np.argsort(dist)[:inter_knn]:
            query_reference_edges[(cid, ref_feat_df.index[i])] = dist[i]

    n_edges = int(ratio * len(query_reference_edges))
    for e in sorted(query_reference_edges.keys(), key=lambda x: query_reference_edges[x])[:n_edges]:
        query_reference_neighbor_df[e[0]].append(e[1])
    query_reference_neighbor_df = pd.DataFrame(
        {"query-reference feature": [query_reference_neighbor_df[cid] for cid in query_feat_df.index]},
        index=query_feat_df.index)
    return query_reference_neighbor_df


def SCGPExtension_wrapper(query_objs,
                          ref_dfs,
                          exclude_features=[],
                          feature_weights={},
                          seed=122,
                          verbose=True,
                          delaunay_distance_cutoff=35,
                          pixel_resolution=0.3775,
                          rp=1e-3,
                          intra_feature_knn=3,
                          inter_feature_knn=3,
                          ratio=0.5,
                          smooth_level=0,
                          smooth_iter=1,
                          feature_item={'expression': 1},
                          attach_to_object=True):
    """ Wrapper function for SCGP-Extension

    Args:
        query_objs (AnnData/EMObject/list): (list of) query/unseen region object(s)
        ref_dfs (tuple): a tuple of three pseudo nodes pd.DataFrame, see
            outputs of functions `select_pseudo_nodes` and `make_pseudo_nodes`.
        exclude_features (list, optional): list of features to exclude.
        feature_weights (dict, optional): dict of feature weights.
            See `features.reweigh_features` for details.
        seed (int, optional): random seed. Defaults to 123.
        verbose (bool, optional): verbosity. Defaults to True.
        delaunay_distance_cutoff (float, optional): distance cutoff (in um) for
            excluding distant delaunay triangulation edges. Defaults to 35.
        pixel_resolution (float, optional): length (in um) per pixel.
        rp (float, optional): resolution parameter for leiden partition.
        intra_feature_knn (int, optional): number of nearest neighbors to consider
            for feature edges between query nodes.
        inter_feature_knn (int, optional): number of nearest neighbors to consider
            for reference-query feature edges.
        ratio (float, optional): subsample ratio of reference-query feature edges,
            these edges will be ranked based on similarity and the `ratio` most
            similar edges will be kept. Should be between 0 and 1.
        smooth_level (int, optional): smooth level for post partition
            smoothing, see `scgp.partition.smooth_spatially_isolated_patch`.
        smooth_iter (int, optional): number of smoothing runs.
        feature_item (dict, optional): feature item to calculate. Use empty dict
            to use existing feature dataframe. Defaults to {'expression': 1}.
        attach_to_object (bool, optional): if to attach the resulting SCGPExt
            partitions to the input region object(s).

    Returns:
        dict: dict of cell(node) name: SCGP-Extension partition id.
        tuple: tuple of features and model used for reproduction or finetuning.
    """
    query_objs = [query_objs] if not isinstance(query_objs, list) else query_objs
    bms = get_biomarkers(query_objs[0])
    assert [get_biomarkers(obj) == bms for obj in query_objs]

    # Query feature and query spatial neighbors
    t0 = time.time()
    query_feat_df = []
    query_spatial_neighbor_df = []
    for obj in query_objs:
        build_delaunay_triangulation_neighborhood(
            obj, r=delaunay_distance_cutoff, um_per_px=pixel_resolution, attach_to_object=True)
        if len(feature_item) > 0:
            calculate_feature(obj, feature_item=feature_item)
        # Extract feature and neighborhood, with rows indexed by (region_id, cell_id)
        feature_df, neighbor_df = extract_feature_neighborhood_with_region_cell_ids(obj)
        feature_df = reweigh_features(
            feature_df,
            exclude_features=exclude_features,
            feature_weights=feature_weights)
        query_feat_df.append(feature_df)
        query_spatial_neighbor_df.append(neighbor_df)
    query_feat_df = pd.concat(query_feat_df, axis=0)
    query_spatial_neighbor_df = pd.concat(query_spatial_neighbor_df, axis=0)

    # Query feature edges
    query_feature_neighbor_df = build_feature_knn_umap(
        query_feat_df, k=intra_feature_knn, seed=seed, attach_to_object=False)

    # Reference-query feature edges
    pn_feat_df, pn_membership_df, pn_neighborhood_df = ref_dfs
    query_reference_neighbor_df = build_sample_reference_edges(
        query_feat_df, pn_feat_df, inter_knn=inter_feature_knn, ratio=ratio)

    # Combined neighborhood dataframe
    query_neighborhood_df = pd.concat([
        query_spatial_neighbor_df,
        query_feature_neighbor_df,
        query_reference_neighbor_df], axis=1)
    pn_neighborhood_df['spatial'] = [[]] * pn_neighborhood_df.shape[0]
    pn_neighborhood_df['query-reference feature'] = [[]] * pn_neighborhood_df.shape[0]
    combined_neighborhood_df = pd.concat([query_neighborhood_df, pn_neighborhood_df], axis=0)
    if verbose:
        n_spatial_edges = count_edges_in_neighborhood(query_spatial_neighbor_df)
        n_feature_edges = count_edges_in_neighborhood(query_feature_neighbor_df)
        n_query_reference_edges = count_edges_in_neighborhood(query_reference_neighbor_df)
        print("Use %d spatial edges, %d feature edges, %d query-reference feature edges" %
              (n_spatial_edges, n_feature_edges, n_query_reference_edges))

    # Combined feature dataframe
    combined_feat_df = pd.concat([query_feat_df, pn_feat_df], axis=0)
    t_f = time.time()

    # Combined cellular graph
    joint_graph = construct_graph(
        combined_neighborhood_df, feature_df=combined_feat_df, weighted=True, normalize=True)

    # Partition initialization
    is_membership_fixed = [0] * query_feat_df.shape[0] + [1] * pn_feat_df.shape[0]
    initial_membership = map_to_pseudo_nodes(
        query_feat_df, ref_dfs, knn=5) + list(np.array(pn_membership_df).flatten())
    
    # Run partition
    feats = (query_feat_df, query_spatial_neighbor_df)
    model = (joint_graph, is_membership_fixed, initial_membership)
    query_gp = SCGPExtension_partition(
        feats, model, rp=rp, smooth_level=smooth_level, smooth_iter=smooth_iter, verbose=verbose)

    t_c = time.time()
    if verbose:
        print("Featurization takes %.2fs, Clustering takes %.2fs" % (t_f - t0, t_c - t_f))

    if attach_to_object:
        assign_annotation_dict_to_objects(
            query_gp, query_objs, name='SCGPExt_annotation', categorical=True)
    return query_gp, (feats, model)


def SCGPExtension_partition(feats,
                            model,
                            rp=1e-3,
                            smooth_level=0,
                            smooth_iter=1,
                            verbose=False):
    """Partitioning the joint graph defined by SCGPExt

    Args:
        feats (tuple): tuple of feature and neighborhood data frames.
        model (tuple): the joint graph of query nodes and pseudo-nodes from reference.
        rp (float, optional): resolution parameter for leiden partition.
        smooth_level (int, optional): smooth level for post partition
            smoothing, see `scgp.partition.smooth_spatially_isolated_patch`.
        smooth_iter (int, optional): number of smoothing runs.

    Returns:
        dict: dict of cell(node) name: SCGP-Extension partition id.
    """
    
    query_feat_df, query_spatial_neighbor_df = feats
    joint_graph, is_membership_fixed, initial_membership = model

    combined_membership = leiden_partition_with_reference(
        joint_graph, initial_membership=initial_membership, is_membership_fixed=is_membership_fixed, rp=rp)
    query_membership = combined_membership[:query_feat_df.shape[0]]

    query_membership = remove_isolated_patch(query_membership)
    for _ in range(smooth_iter):
        query_membership = smooth_spatially_isolated_patch(
            query_spatial_neighbor_df, query_membership, smooth_level=smooth_level)
    query_membership = remove_isolated_patch(query_membership)
    if verbose:
        print("Find %d partitions" % (len(set(query_membership)) - 1))  # Remove unknown

    query_gp = membership_to_membership_dict(query_membership, query_feat_df)
    return query_gp
