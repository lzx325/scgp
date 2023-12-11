import os
import numpy as np
import pandas as pd
from copy import deepcopy

from scgp.object_io import (
    load_cell_coords,
    load_cell_biomarker_expression,
    construct_object,
    get_name,
    get_biomarkers,
    get_cell_ids,
    get_cell_positions,
    get_raw_biomarker_expression,
    get_normed_biomarker_expression,
    get_cell_neighborhood,
    get_cell_annotation,
    has_cell_annotation,
    get_feature,
    assign_annotation,
    assign_annotation_dict_to_objects,
)
from scgp.neighborhood import (
    build_distance_neighborhood,
    build_delaunay_triangulation_neighborhood,
    build_knn_neighborhood,
    build_feature_knn_kdtree,
    build_feature_knn_umap,
)
from scgp.features import (
    get_unique_cell_types,
    get_local_count,
    get_local_composition,
    get_local_density,
    get_local_smoothed_biomarker_expression,
    calculate_feature,
)
from scgp.scgp_wrapper import (
    biomarker_clustering_wrapper,
    cellular_neighborhood_wrapper,
    UTAG_wrapper,
    SCGP_wrapper,
)
from scgp.scgp_extension import (
    select_pseudo_nodes,
    make_pseudo_nodes,
    SCGPExtension_wrapper,
)

test_regions = [
    's255_c001_v001_r001_reg002',
    's255_c001_v001_r001_reg003',
    's255_c001_v001_r001_reg004',
    's255_c001_v001_r001_reg005',
]


def test_construct_objects():
    region_id = test_regions[0]
    cell_seg_df = load_cell_coords('data/DKD_Kidney/%s.cell_data.csv' % region_id)
    biomarker_expression_df = load_cell_biomarker_expression('data/DKD_Kidney/%s.expression.csv' % region_id)
    cell_annotation_df = deepcopy(cell_seg_df[['CELL_ID']])
    cell_annotation_df['ANN1'] = np.random.randint(0, 3, len(cell_annotation_df))
    cell_annotation_df['ANN2'] = np.random.normal(0, 1, len(cell_annotation_df))

    num_cells = cell_seg_df.shape[0]
    num_bms = biomarker_expression_df.shape[1] - 1

    for mode in ['anndata', 'emobject']:
        obj = construct_object(
            region_id,
            cell_seg_df,
            biomarker_expression_df,
            cell_annotation_df=cell_annotation_df,
            index_col='CELL_ID',
            mode=mode)
        assert get_name(obj) == region_id
        assert set(get_cell_ids(obj)) == set([str(cid) for cid in cell_seg_df['CELL_ID']])
        assert set(get_biomarkers(obj)) == set(biomarker_expression_df.columns) - set(['CELL_ID'])

        assert get_raw_biomarker_expression(obj).shape == (num_cells, num_bms)
        assert np.min(np.array(get_raw_biomarker_expression(obj))) >= 0

        assert get_normed_biomarker_expression(obj).shape == (num_cells, num_bms)
        assert np.allclose(np.array(get_normed_biomarker_expression(obj)).mean(0), 0)

        assert get_cell_positions(obj).shape == (num_cells, 2)
        assert np.issubdtype(np.array(get_cell_positions(obj)).dtype, np.integer)
        assert np.min(np.array(get_cell_positions(obj))) >= 0

        assert get_cell_neighborhood(obj).shape == (num_cells, 0)

        assert get_cell_annotation(obj, annotation_name='ANN1').shape == (num_cells, 1)
        assert get_cell_annotation(obj).shape == (num_cells, 2)
        assert has_cell_annotation(obj, annotation_name='ANN2')

        new_annotation = cell_annotation_df[['CELL_ID']]
        new_annotation = new_annotation.set_index('CELL_ID')
        new_annotation['N_ANN1'] = np.random.randint(0, 3, len(new_annotation))
        assign_annotation(obj, new_annotation, name='ANN2')
        assert np.allclose(get_cell_annotation(obj, annotation_name='ANN2'), new_annotation[['N_ANN1']])

        new_annotation_dict = {(get_name(obj), cid): np.random.uniform() for cid in get_cell_ids(obj)}
        assign_annotation_dict_to_objects(new_annotation_dict, [obj], name='N_ANN2')
        assert np.allclose(get_cell_annotation(obj, annotation_name='N_ANN2'),
                           pd.DataFrame(new_annotation_dict.values(),
                                        index=new_annotation_dict.keys(),
                                        columns=['N_ANN2']))


def test_construct_spatial_neighborhood():
    region_id = test_regions[1]
    cell_seg_df = load_cell_coords('data/DKD_Kidney/%s.cell_data.csv' % region_id)
    biomarker_expression_df = load_cell_biomarker_expression('data/DKD_Kidney/%s.expression.csv' % region_id)

    for mode in ['anndata', 'emobject']:
        obj = construct_object(
            region_id,
            cell_seg_df,
            biomarker_expression_df,
            cell_annotation_df=None,
            index_col='CELL_ID',
            mode=mode)

        for find_neighbor_fn in [
                build_distance_neighborhood,
                build_knn_neighborhood,
                build_delaunay_triangulation_neighborhood]:
            neighborhood = find_neighbor_fn(obj)
            find_neighbor_fn(obj, attach_to_object=True)
            assert np.all(np.array(neighborhood) == np.array(get_cell_neighborhood(obj, neighbor_type='spatial')))


def test_calculate_features():
    region_id = test_regions[2]
    cell_seg_df = load_cell_coords('data/DKD_Kidney/%s.cell_data.csv' % region_id)
    biomarker_expression_df = load_cell_biomarker_expression('data/DKD_Kidney/%s.expression.csv' % region_id)
    pseudo_cell_type_df = deepcopy(cell_seg_df[['CELL_ID']])
    pseudo_cell_type_df['cell_type'] = np.random.randint(0, 10, len(pseudo_cell_type_df))
    for mode in ['anndata', 'emobject']:
        obj = construct_object(
            region_id,
            cell_seg_df,
            biomarker_expression_df,
            cell_annotation_df=pseudo_cell_type_df,
            index_col='CELL_ID',
            mode=mode)
        build_delaunay_triangulation_neighborhood(obj, attach_to_object=True)

        unique_cell_types = get_unique_cell_types(obj)
        assert len(set(unique_cell_types) - set(range(10))) == 0

        feat = get_local_count(obj, unique_cell_types)
        assert feat.shape == (len(get_cell_ids(obj)), len(unique_cell_types))
        assert [col.startswith('Feat-Count-') for col in feat.columns]

        feat = get_local_composition(obj, unique_cell_types)
        assert feat.shape == (len(get_cell_ids(obj)), len(unique_cell_types))
        assert [col.startswith('Feat-Composition-') for col in feat.columns]
        assert np.allclose(np.array(feat.sum(1)), 1)

        feat = get_local_density(obj)
        assert feat.shape == (len(get_cell_ids(obj)), 3)
        assert [col.startswith('Feat-Distance-') for col in feat.columns]

        feat = get_local_smoothed_biomarker_expression(obj)
        assert feat.shape == (len(get_cell_ids(obj)), len(get_biomarkers(obj)))
        assert [col == 'Feat-Smoothed_Expression-%s' % bm for col, bm in zip(feat.columns, get_biomarkers(obj))]


def test_construct_features():
    region_id = test_regions[3]
    cell_seg_df = load_cell_coords('data/DKD_Kidney/%s.cell_data.csv' % region_id)
    biomarker_expression_df = load_cell_biomarker_expression('data/DKD_Kidney/%s.expression.csv' % region_id)
    pseudo_cell_type_df = deepcopy(cell_seg_df[['CELL_ID']])
    pseudo_cell_type_df['cell_type'] = np.random.randint(0, 10, len(pseudo_cell_type_df))
    for mode in ['anndata', 'emobject']:
        obj = construct_object(
            region_id,
            cell_seg_df,
            biomarker_expression_df,
            cell_annotation_df=pseudo_cell_type_df,
            index_col='CELL_ID',
            mode=mode)
        feature_item = {
            'expression': 1.,
            'smoothed_expression': 1.,
            'composition': 1.,
            'count': 0.1,
            'entropy': 1.,
            'coordinate': 0.1,
            'density': 1.,
        }
        unique_cell_types = get_unique_cell_types(obj)

        build_delaunay_triangulation_neighborhood(obj, attach_to_object=True)
        feat_df = calculate_feature(obj, feature_item=feature_item, unique_cell_types=unique_cell_types)
        feat_df_saved = get_feature(obj)
        assert feat_df.shape == (
            len(get_cell_ids(obj)),
            2 * len(get_biomarkers(obj)) + 2 * len(unique_cell_types) + 6)
        assert np.allclose(feat_df, feat_df_saved)

        build_distance_neighborhood(obj, attach_to_object=True)
        feat_df2 = calculate_feature(obj, feature_item=feature_item, unique_cell_types=unique_cell_types)
        feat_df2_saved = get_feature(obj)
        assert np.allclose(feat_df2, feat_df2_saved)
        assert not np.allclose(feat_df_saved, feat_df2_saved)


def test_construct_feature_neighborhood():
    region_id = test_regions[0]
    cell_seg_df = load_cell_coords('data/DKD_Kidney/%s.cell_data.csv' % region_id)
    biomarker_expression_df = load_cell_biomarker_expression('data/DKD_Kidney/%s.expression.csv' % region_id)
    for mode in ['anndata', 'emobject']:
        obj = construct_object(
            region_id,
            cell_seg_df,
            biomarker_expression_df,
            cell_annotation_df=None,
            index_col='CELL_ID',
            mode=mode)
        feature_item = {
            'expression': 1.,
            'smoothed_expression': 1.,
        }
        build_delaunay_triangulation_neighborhood(obj, attach_to_object=True)
        feat_df = calculate_feature(obj, feature_item=feature_item, unique_cell_types=None)
        build_feature_knn_kdtree(obj, k=7, attach_to_object=True)
        assert get_cell_neighborhood(obj).shape == (len(get_cell_ids(obj)), 2)
        assert get_cell_neighborhood(obj, neighbor_type='feature').shape == \
            get_cell_neighborhood(obj, neighbor_type='spatial').shape
        for cid, ns in get_cell_neighborhood(obj, neighbor_type='feature').iterrows():
            assert isinstance(cid, str)
            assert [isinstance(n, str) for n in ns.item()]
            assert len(ns.item()) == 7
            assert cid not in ns.item()

        feat_df_rep = calculate_feature(obj, feature_item=feature_item, unique_cell_types=None)
        assert np.allclose(feat_df, feat_df_rep)

        build_feature_knn_umap(obj, k=7, seed=123, attach_to_object=True)
        assert get_cell_neighborhood(obj).shape == (len(get_cell_ids(obj)), 2)
        assert get_cell_neighborhood(obj, neighbor_type='feature').shape == \
            get_cell_neighborhood(obj, neighbor_type='spatial').shape

        for cid, ns in get_cell_neighborhood(obj, neighbor_type='feature').iterrows():
            assert isinstance(cid, str)
            assert [isinstance(n, str) for n in ns.item()]
            assert cid not in ns.item()


def load_test_objects(mode='anndata'):
    objs = []
    for region_id in test_regions:
        cell_seg_df = load_cell_coords('data/DKD_Kidney/%s.cell_data.csv' % region_id)
        biomarker_expression_df = load_cell_biomarker_expression('data/DKD_Kidney/%s.expression.csv' % region_id)
        obj = construct_object(region_id, cell_seg_df, biomarker_expression_df, index_col='CELL_ID', mode=mode)
        objs.append(obj)
    return objs


def test_biomarker_clustering():
    objs = load_test_objects()
    bm_cls, _ = biomarker_clustering_wrapper(objs)
    assert len(bm_cls) == sum([len(get_cell_ids(obj)) for obj in objs])
    assert bm_cls.keys() == set((get_name(obj), cid) for obj in objs for cid in get_cell_ids(obj))
    assert [isinstance(i, int) for i in bm_cls.values()]


def test_cellular_neighborhood():
    objs = load_test_objects()
    for obj in objs:
        pseudo_cell_type_df = pd.DataFrame(
            {'cell_type': np.random.randint(0, 10, len(get_cell_ids(obj)))},
            index=get_cell_ids(obj))
        assign_annotation(obj, pseudo_cell_type_df, name='cell_type')
    cns, _ = cellular_neighborhood_wrapper(objs, unique_cell_types=list(range(10)))
    assert len(cns) == sum([len(get_cell_ids(obj)) for obj in objs])
    assert cns.keys() == set((get_name(obj), cid) for obj in objs for cid in get_cell_ids(obj))
    assert [isinstance(i, int) for i in cns.values()]


def test_utag():
    objs = load_test_objects()
    utags, _ = UTAG_wrapper(objs)
    assert len(utags) == sum([len(get_cell_ids(obj)) for obj in objs])
    assert utags.keys() == set((get_name(obj), cid) for obj in objs for cid in get_cell_ids(obj))
    assert [isinstance(i, int) for i in utags.values()]


def test_SCGP():
    objs = load_test_objects()
    scgps, _ = SCGP_wrapper(objs)
    assert len(scgps) == sum([len(get_cell_ids(obj)) for obj in objs])
    assert scgps.keys() == set((get_name(obj), cid) for obj in objs for cid in get_cell_ids(obj))
    assert [isinstance(i, int) for i in scgps.values()]


def test_pseudo_nodes():
    objs = load_test_objects()
    scgps, _ = SCGP_wrapper(objs[:1], rp=1e-3)
    use_partitions = set(v for v in scgps.values() if v >= 0)  # -1 is unknown

    for pn_fn in [select_pseudo_nodes, make_pseudo_nodes]:
        pn_feat_df, pn_membership_df, pn_neighbor_df = pn_fn(
            objs[:1], scgps, use_partitions=use_partitions, k=20, intra_knn=4)
        assert pn_feat_df.shape[1] == get_feature(objs[0]).shape[1]
        assert [ind[0].startswith('PN-') for ind in pn_feat_df.index]
        assert [ind[1].startswith('PN-') for ind in pn_feat_df.index]
        assert all(pn_feat_df.index == pn_membership_df.index)
        assert all(pn_feat_df.index == pn_neighbor_df.index)
        assert set(np.array(pn_membership_df).flatten()) == use_partitions
        assert set(sum(list(pn_neighbor_df['feature']), [])) <= set(list(pn_feat_df.index))


def test_SCGPExtension():
    objs = load_test_objects()
    ref_objs = objs[:1]
    query_objs = objs[1:]
    scgps, _ = SCGP_wrapper(ref_objs, rp=1e-3)
    use_partitions = set(v for v in scgps.values() if v >= 0)  # -1 is unknown
    ref_dfs = make_pseudo_nodes(ref_objs, scgps, use_partitions=use_partitions, k=20, intra_knn=4)

    query_scgps, _ = SCGPExtension_wrapper(
        query_objs, ref_dfs, rp=1e-3,
        intra_feature_knn=5, inter_feature_knn=4, ratio=0.5)
    assert len(query_scgps) == sum([len(get_cell_ids(obj)) for obj in query_objs])
    assert query_scgps.keys() == set((get_name(obj), cid) for obj in query_objs for cid in get_cell_ids(obj))
    assert [isinstance(i, int) for i in query_scgps.values()]
