#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 12:39:48 2022

@author: zqwu
"""
import numpy as np
import pandas as pd
from copy import deepcopy

from scgp.neighborhood import build_knn_neighborhood
from scgp.object_io import (
    get_cell_ids,
    get_biomarkers,
    get_cell_positions,
    get_cell_neighborhood,
    has_cell_annotation,
    get_cell_annotation,
    get_normed_biomarker_expression,
    assign_features
)


def get_local_count(obj, unique_cell_types):
    """Feature: count of cell types in the 1-hop neighborhood

    Args:
        obj (AnnData/EMObject): region object
        unique_cell_types (list): list of unique cell types

    Returns:
        pd.DataFrame: feature dataframe
    """
    assert has_cell_annotation(obj, 'cell_type')
    unique_cell_types = list(unique_cell_types)
    count_vec = np.zeros((len(get_cell_ids(obj)), len(unique_cell_types)))
    cell_type_serie = get_cell_annotation(obj, 'cell_type').squeeze()
    spatial_neighborhood_serie = get_cell_neighborhood(obj, neighbor_type='spatial').squeeze()
    for i, n in enumerate(spatial_neighborhood_serie):
        neighbor_cts = cell_type_serie.loc[n]
        for n_ct in neighbor_cts:
            count_vec[i, unique_cell_types.index(n_ct)] += 1
    count_vec = pd.DataFrame(
        count_vec,
        index=get_cell_ids(obj),
        columns=['Feat-Count-%s' % str(ct) for ct in unique_cell_types])
    return count_vec


def get_local_composition(obj, unique_cell_types):
    """Feature: composition (frequency) of cell types in the 1-hop neighborhood

    Args:
        obj (AnnData/EMObject): region object
        unique_cell_types (list): list of unique cell types

    Returns:
        pd.DataFrame: feature dataframe
    """
    count_vec = get_local_count(obj, unique_cell_types)
    composition_vec = count_vec / (1e-5 + np.array(count_vec.sum(1)).reshape((-1, 1)))
    composition_vec.columns = [
        col.replace('Feat-Count', 'Feat-Composition') for col in composition_vec.columns]
    return composition_vec


def get_local_smoothed_biomarker_expression(obj):
    """Feature: 1-hop smoothed biomarker expression

    Args:
        obj (AnnData/EMObject): region object

    Returns:
        pd.DataFrame: feature dataframe
    """
    exp_mat = get_normed_biomarker_expression(obj)
    smoothed_exp_ar = np.zeros_like(exp_mat)
    spatial_neighborhood_serie = get_cell_neighborhood(obj, neighbor_type='spatial').squeeze()
    for i, n in enumerate(spatial_neighborhood_serie):
        neighbor_exp = exp_mat.loc[n]
        smoothed_exp_ar[i] = np.mean(neighbor_exp, 0)
    smoothed_exp_ar = pd.DataFrame(
        smoothed_exp_ar,
        index=get_cell_ids(obj),
        columns=['Feat-Smoothed_Expression-%s' % bm for bm in get_biomarkers(obj)])
    return smoothed_exp_ar


def get_local_density(obj):
    """Feature: cell density, calcualted as median distance to k-nearest neighbor cells
        k = 6 ~ 1-hop
        k = 20 ~ 2-hop
        k = 40 ~ 3-hop

    Args:
        obj (AnnData/EMObject): region object

    Returns:
        pd.DataFrame: feature dataframe
    """
    _, knn_distances = build_knn_neighborhood(
        obj, k=41, return_neighbor_distances=True, attach_to_object=False)

    data = np.stack([np.median(knn_distances[:, 1:7], 1),
                     np.median(knn_distances[:, 1:21], 1),
                     np.median(knn_distances[:, 1:41], 1)], 1)
    local_density = pd.DataFrame(
        data,
        index=get_cell_ids(obj),
        columns=['Feat-Distance-6nn', 'Feat-Distance-20nn', 'Feat-Distance-40nn'])
    return local_density


def get_unique_cell_types(obj):
    """Get unique cell types from the region

    Args:
        obj (AnnData/EMObject): region object

    Returns:
        list: list of unique cell types
    """
    if not has_cell_annotation(obj, 'cell_type'):
        return []
    else:
        cell_types = get_cell_annotation(obj, 'cell_type').squeeze()
        return sorted(set(np.array(cell_types)))


def calculate_feature(obj, feature_item={'expression': 1.}, unique_cell_types=None):
    """Calculate feature dataframe for the region

    Support the features below:
    [
        'expression',
        'smoothed_expression',
        'composition',
        'count',
        'entropy',
        'coordinate',
        'density',
    ]

    Args:
        obj (AnnData/EMObject): region object
        feature_item (dict): dict of feature items and weights, if multiple
            feature items are provided, the final feature dataframe will be
            the concatenation of each weighted feature dataframes
        unique_cell_types (list): list of unique cell types

    Returns:
        pd.DataFrame: combined feature dataframe
    """
    unique_cell_types = unique_cell_types if unique_cell_types is not None else get_unique_cell_types(obj)

    features = []
    for key, w in feature_item.items():
        if key == 'expression':
            exp_df = deepcopy(get_normed_biomarker_expression(obj))
            exp_df.columns = ['Feat-Expression-%s' % bm for bm in get_biomarkers(obj)]
            features.append(exp_df * w)
        elif key == 'smoothed_expression':
            smoothed_exp_df = get_local_smoothed_biomarker_expression(obj)
            features.append(smoothed_exp_df * w)
        elif key == 'count':
            count_vec_df = get_local_count(obj, unique_cell_types)
            features.append(count_vec_df * w)
        elif key == 'composition':
            composition_vec_df = get_local_composition(obj, unique_cell_types)
            features.append(composition_vec_df * w)
        elif key == 'entropy':
            composition_vec_df = get_local_composition(obj, unique_cell_types)
            entropy = -(composition_vec_df * np.log(composition_vec_df + 1e-5)).sum(1)
            entropy = entropy.to_frame(name='Feat-Entropy')
            features.append(entropy * w)
        elif key == 'coordinate':
            coords = deepcopy(get_cell_positions(obj))
            coords.columns = ['Feat-Coordinate-%s' % c for c in coords.columns]
            features.append(coords * w)
        elif key == 'density':
            local_density_df = get_local_density(obj)
            features.append(local_density_df * w)
        else:
            raise ValueError("feature item not recognized")
    assert all([f.shape[0] == features[0].shape[0] for f in features])
    features = pd.concat(features, axis=1)
    assign_features(obj, features)
    return features


def reweigh_features(feature_df, exclude_features=[], feature_weights={}):
    """ Assign weights to features

    Args:
        feature_df (pd.DataFrame): feature dataframe
        exclude_features (list): list of features to exclude
        feature_weights (dict): dict of feature weights, feature not included
            in this dict will be assigned default weights of 1.

    Returns:
        pd.DataFrame: reweighted feature dataframe
    """
    feature_cols = feature_df.columns
    _f_w = {f: 1. for f in feature_cols}
    for f in exclude_features:
        _f_w[f] = 0
    _f_w.update(feature_weights)

    weight_array = np.array([_f_w[f] for f in feature_cols])
    feature_df = feature_df * weight_array.reshape((1, -1))
    return feature_df
