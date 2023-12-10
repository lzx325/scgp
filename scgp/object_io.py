import numpy as np
import pandas as pd
import scipy
import warnings
from copy import deepcopy


def load_cell_coords(cell_coords_file):
    """Load cell coordinates from file

    Args:
        cell_coords_file (str): path to csv file containing cell coordinates

    Returns:
        pd.DataFrame: dataframe containing cell coordinates, columns ['CELL_ID', 'X', 'Y']
    """
    df = pd.read_csv(cell_coords_file)
    assert 'X' in df.columns, "Cannot find column for X coordinates"
    assert 'Y' in df.columns, "Cannot find column for Y coordinates"
    if 'CELL_ID' not in df.columns:
        warnings.warn("Cannot find column for cell id, using index as cell id")
        df['CELL_ID'] = df.index
    return df[['CELL_ID', 'X', 'Y']]


def load_cell_biomarker_expression(cell_biomarker_expression_file):
    """Load cell biomarker expression from file

    Args:
        cell_biomarker_expression_file (str): path to csv file containing cell biomarker expression

    Returns:
        pd.DataFrame: dataframe containing cell biomarker expression,
            columns ['CELL_ID', '<biomarker1_name>', '<biomarker2_name>', ...]
    """
    df = pd.read_csv(cell_biomarker_expression_file)
    biomarkers = sorted([c for c in df.columns if c != 'CELL_ID'])
    for bm in biomarkers:
        if df[bm].dtype not in [np.dtype(int), np.dtype(float), np.dtype('float64')]:
            warnings.warn("Skipping column %s as it is not numeric" % bm)
            biomarkers.remove(bm)

    if 'CELL_ID' not in df.columns:
        warnings.warn("Cannot find column for cell id, using index as cell id")
        df['CELL_ID'] = df.index
    return df[['CELL_ID'] + biomarkers]


def normalize_biomarker_expression(bm_exp_df):
    """ Normalize biomarker expression table

    Only support arcsinh-zscore pipeline for now

    Args:
        bm_exp_df (pd.DataFrame): dataframe of raw biomarker expression

    Returns:
        pd.DataFrame: dataframe of normalized biomarker expression
    """
    assert 'CELL_ID' not in bm_exp_df.columns
    bm_exp_df = np.arcsinh(bm_exp_df / (5 * np.quantile(bm_exp_df, 0.2, axis=0) + 1e-5))
    bm_exp_df = (bm_exp_df - bm_exp_df.mean(0)) / bm_exp_df.std(0)
    return bm_exp_df


def load_cell_annotations(cell_annotations_file, annotation_column='CELL_TYPE'):
    """Load cell annotations from file

    Args:
        cell_annotations_file (str): path to csv file containing cell annotations

    Returns:
        pd.DataFrame: dataframe containing cell annotations, columns ['CELL_ID', 'CELL_TYPE']
    """
    df = pd.read_csv(cell_annotations_file)
    assert annotation_column in df.columns, "Cannot find column %s for cell annotation" % annotation_column

    if 'CELL_ID' not in df.columns:
        warnings.warn("Cannot find column for cell id, using index as cell id")
        df['CELL_ID'] = df.index
    return df[['CELL_ID', annotation_column]]


def construct_object(
        region_id,
        cell_seg_df,
        biomarker_expression_df,
        cell_annotation_df=None,
        normalize_fn=normalize_biomarker_expression,
        index_col='CELL_ID',
        mode='emobject'):
    """Construct an Annotated Data Matrix

    Args:
        region_id (str): name of the region
        cell_seg_df (pd.DataFrame): cell centroid coordinates, should contain
            a cell index column and the following columns: 'X', 'Y'
        biomarker_expression_df (pd.DataFrame): main biomarker expression
            matrix, should contain a cell index column
        cell_annotation_df (pd.DataFrame, optional): cell annotations/cell types,
            should contain a cell index column
        normalize_fn (function, optional): function to normalize biomarker expression.
            Use None if biomarker expression matrix is already normalized
        index_col (str, optional): column name for the cell index column.
            Defaults to 'CELL_ID'.
        mode (str, optional): mode of the object, one of 'emobject' or 'anndata'

    Returns:
        AnnData: region object
    """
    assert index_col in biomarker_expression_df
    assert index_col in cell_seg_df
    
    # cell index with string type
    cell_seg_df[index_col] = cell_seg_df[index_col].astype(str)
    biomarker_expression_df[index_col] = biomarker_expression_df[index_col].astype(str)
    assert len(biomarker_expression_df[index_col]) == len(set(biomarker_expression_df[index_col])), \
        "Duplicate indexes found in biomarker expression matrix"
    assert sorted(biomarker_expression_df[index_col]) == sorted(cell_seg_df[index_col]), \
        "Cell indexes in biomarker expression df and cell segmentation df do not match"
    indices = list(biomarker_expression_df[index_col])
    columns = [c for c in biomarker_expression_df.columns if c != index_col]
    assert all(isinstance(i, str) for i in indices)
    assert all(isinstance(c, str) for c in columns)

    bm_exp_df = biomarker_expression_df.set_index(index_col)
    bm_exp_df = bm_exp_df[columns].loc[indices]
    coord_df = cell_seg_df.set_index(index_col)
    coord_df = coord_df[['X', 'Y']].loc[indices]

    if cell_annotation_df is not None:
        assert index_col in cell_annotation_df
        cell_annotation_df[index_col] = cell_annotation_df[index_col].astype(str)
        assert sorted(cell_annotation_df.index) == sorted(biomarker_expression_df.index), \
            "Cell indexes in biomarker expression df and cell annotation df do not match"
        assert "X" not in cell_annotation_df.columns
        assert "Y" not in cell_annotation_df.columns

        ann_df = cell_annotation_df.set_index(index_col)
        ann_df = ann_df.loc[indices]
    
    if mode == 'anndata':
        import anndata as ad
        X = np.array(bm_exp_df)
        adata = ad.AnnData(X, dtype=float)
        adata.obs_names = indices
        adata.var_names = columns
        if normalize_fn is None:
            normalize_fn = lambda x: x
        adata.layers["normalized"] = np.array(normalize_fn(bm_exp_df))
        adata.obs["X"] = list(coord_df.loc[indices, "X"])
        adata.obs["Y"] = list(coord_df.loc[indices, "Y"])
        adata.uns["name"] = region_id
        adata.obsm["spatial"] = np.array(coord_df.loc[indices, ["X", "Y"]])
        if cell_annotation_df is not None:
            for col in ann_df.columns:
                adata.obs[col] = list(ann_df.loc[indices, col])
        return adata
    elif mode == 'emobject':
        from emobject.emobject import EMObject
        from emobject.emlayer import BaseLayer
        emobj = EMObject(
            data=bm_exp_df,
            obs=bm_exp_df[[]],
            pos=coord_df,
            name=region_id,
            first_layer_name='raw')
        normed_data = normalize_fn(bm_exp_df)
        normed_layer = BaseLayer(data=normed_data, name='normalized')
        emobj.add(normed_layer)
        emobj.set_layer('raw')
        if cell_annotation_df is not None:
            for col in ann_df.columns:
                emobj.add_anno(attr='obs', value=list(ann_df[col]), name=col)
        emobj.neighbors = bm_exp_df[[]]
        return emobj
    else:
        raise ValueError("Unknown mode %s" % mode)


# Access data from EMObject/AnnData
def get_name(obj):
    """Get name of the region

    Args:
        obj (AnnData/EMObject): region object

    Returns:
        str: name of the region
    """
    if type(obj).__name__ == 'AnnData':
        return obj.uns['name']
    elif type(obj).__name__ == 'EMObject':
        return obj.name
    else:
        raise ValueError('Input format %s not recognized' % str(type(obj)))


def get_cell_ids(obj):
    """Get list of cell/observation names in the region

    Args:
        obj (AnnData/EMObject): region object

    Returns:
        list: list of cell/observation names
    """
    if type(obj).__name__ == 'AnnData':
        return list(obj.obs_names)
    elif type(obj).__name__ == 'EMObject':
        obj.set_layer('raw')
        return list(obj.obs_ax)
    else:
        raise ValueError('Input format %s not recognized' % str(type(obj)))


def get_biomarkers(obj):
    """Get list of biomarker/variable names in the region

    Args:
        obj (AnnData/EMObject): region object

    Returns:
        list: list of biomarker/variable names
    """
    if type(obj).__name__ == 'AnnData':
        return list(obj.var_names)
    elif type(obj).__name__ == 'EMObject':
        obj.set_layer('raw')
        return list(obj.var_ax)
    else:
        raise ValueError('Input format %s not recognized' % str(type(obj)))


def get_raw_biomarker_expression(obj):
    """Get raw biomarker expression from the region

    Args:
        obj (AnnData/EMObject): region object

    Returns:
        pd.DataFrame: dataframe of raw biomarker expression
    """
    if type(obj).__name__ == 'AnnData':
        exp_ar = obj.X
        exp_df = pd.DataFrame(data=exp_ar, index=get_cell_ids(obj), columns=get_biomarkers(obj))
        return exp_df
    elif type(obj).__name__ == 'EMObject':
        obj.set_layer('raw')
        return obj.data
    else:
        raise ValueError('Input format %s not recognized' % str(type(obj)))


def get_normed_biomarker_expression(obj):
    """Get normalized biomarker expression from the region

    Args:
        obj (AnnData/EMObject): region object

    Returns:
        pd.DataFrame: dataframe of normalized biomarker expression
    """
    if type(obj).__name__ == 'AnnData':
        exp_ar = obj.layers["normalized"]
        exp_df = pd.DataFrame(data=exp_ar, index=get_cell_ids(obj), columns=get_biomarkers(obj))
        return exp_df
    elif type(obj).__name__ == 'EMObject':
        obj.set_layer('normalized')
        return obj.data
    else:
        raise ValueError('Input format %s not recognized' % str(type(obj)))


def get_cell_positions(obj):
    """Get cell positions from the region

    Args:
        obj (AnnData/EMObject): region object

    Returns:
        pd.DataFrame: dataframe of cell positions
    """
    if type(obj).__name__ == 'AnnData':
        return obj.obs[['X', 'Y']]
    elif type(obj).__name__ == 'EMObject':
        obj.set_layer('raw')
        return obj.pos['raw'][['X', 'Y']]
    else:
        raise ValueError('Input format %s not recognized' % str(type(obj)))


def get_cell_neighborhood(obj, neighbor_type=None):
    """Get neighborhood of cells from the region

    Args:
        obj (AnnData/EMObject): region object
        neighbor_type (str): neighbor type, one of "spatial", "feature"

    Returns:
        pd.DataFrame: dataframe of neighbor list
    """
    if type(obj).__name__ == 'AnnData':
        neighbor_cols = [c for c in obj.obs.columns if c.startswith('neighbors-')]
        if neighbor_type is None:
            return obj.obs[neighbor_cols]
        else:
            assert 'neighbors-%s' % neighbor_type in neighbor_cols
            return obj.obs[['neighbors-%s' % neighbor_type]]
    elif type(obj).__name__ == 'EMObject':
        if neighbor_type is None:
            return obj.neighbors
        else:
            assert neighbor_type in obj.neighbors.columns
            return obj.neighbors[[neighbor_type]]
    else:
        raise ValueError('Input format %s not recognized' % str(type(obj)))


def get_cell_annotation(obj, annotation_name=None):
    """Get an annotation from the region

    Args:
        obj (AnnData/EMObject): region object
        annotation_name (str): name of the annotation

    Returns:
        pd.DataFrame: annotation dataframe, single column
    """
    if annotation_name is None:
        annotation_name = [c for c in obj.obs.columns if c not in ['X', 'Y']]
    elif isinstance(annotation_name, str):
        annotation_name = [annotation_name]

    if type(obj).__name__ == 'AnnData':
        return obj.obs[annotation_name]
    elif type(obj).__name__ == 'EMObject':
        obj.set_layer('raw')
        return obj.obs[annotation_name]
    else:
        raise ValueError('Input format %s not recognized' % str(type(obj)))


def has_cell_annotation(obj, annotation_name='cell_type'):
    """Check if the region contains an annotation

    Args:
        obj (AnnData/EMObject): region object
        annotation_name (str): name of the annotation

    Returns:
        bool: if the region object contains the annotation
    """
    if type(obj).__name__ == 'AnnData':
        return annotation_name in obj.obs.columns
    elif type(obj).__name__ == 'EMObject':
        obj.set_layer('raw')
        return annotation_name in obj.obs.columns
    else:
        raise ValueError('Input format %s not recognized' % str(type(obj)))


def has_feature(obj):
    """Check if the region contains a feature dataframe

    Args:
        obj (AnnData/EMObject): region object

    Returns:
        bool: if the region object contains a feature dataframe
    """
    if type(obj).__name__ == 'AnnData':
        return 'features' in obj.obsm.keys()
    elif type(obj).__name__ == 'EMObject':
        return 'features' in obj.layers
    else:
        raise ValueError('Input format %s not recognized' % str(type(obj)))


def get_feature(obj):
    """Get cellular feature dataframe from the region
    See `featurize.calculate_feature` for details

    Args:
        obj (AnnData/EMObject): region object

    Returns:
        pd.DataFrame: feature dataframe
    """
    if type(obj).__name__ == 'AnnData':
        return obj.obsm['features']
    elif type(obj).__name__ == 'EMObject':
        obj.set_layer('features')
        return obj.data
    else:
        raise ValueError('Input format %s not recognized' % str(type(obj)))


# Write attributes to EMObject/AnnData
def assign_neighborhood(obj, neighbor_df, neighbor_type='spatial'):
    """Assign neighborhood dataframe to the region

    Args:
        obj (AnnData/EMObject): region object
        neighbor_df (pd.DataFrame): dataframe of neighbor list, should contain only one column
        neighbor_type (str): neighbor type, one of "spatial", "feature"
    """
    assert sorted(neighbor_df.index) == sorted(get_cell_ids(obj))
    assert neighbor_df.shape[1] == 1
    neighbor_df = neighbor_df.loc[get_cell_ids(obj)]

    if type(obj).__name__ == 'AnnData':
        obj.obs['neighbors-%s' % neighbor_type] = neighbor_df[neighbor_df.columns[0]]

        # For compatibility with squidpy
        row_ind = []
        col_ind = []
        cell_id_mapping = {cid: i for i, cid in enumerate(get_cell_ids(obj))}
        for cid, neighbor_cell_ids in neighbor_df.iterrows():
            for n_cid in neighbor_cell_ids.item():
                row_ind.append(cell_id_mapping[cid])
                col_ind.append(cell_id_mapping[n_cid])

        num_cells = len(get_cell_ids(obj))
        connectivity_mat = scipy.sparse.csr_matrix(
            ([1]*len(row_ind), (row_ind, col_ind)), shape=(num_cells, num_cells))
        obj.obsp['%s_connectivities' % neighbor_type] = connectivity_mat

    elif type(obj).__name__ == 'EMObject':
        neighbor_col_name = neighbor_type
        if hasattr(obj, 'neighbors'):
            obj.neighbors[neighbor_col_name] = neighbor_df[neighbor_df.columns[0]]
        else:
            _ns = neighbor_df.copy()
            _ns.columns = [neighbor_col_name]
            obj.neighbors = _ns
    else:
        warnings.warn("Unknown input format %s, skipping" % str(type(obj)))
    return


def assign_features(obj, feature_df):
    """Assign feature dataframe to the region object

    Args:
        obj (AnnData/EMObject): region object
        feature_df (pd.DataFrame): feature dataframe
    """
    assert sorted(feature_df.index) == sorted(get_cell_ids(obj))
    feature_df = feature_df.loc[get_cell_ids(obj)]

    if type(obj).__name__ == 'AnnData':
        obj.obsm['features'] = feature_df
    elif type(obj).__name__ == 'EMObject':
        from emobject.emlayer import BaseLayer
        feature_layer = BaseLayer(data=feature_df, name='features')
        obj.add(feature_layer)
    else:
        warnings.warn("Unknown input format %s, skipping" % str(type(obj)))
    return


def assign_annotation(obj, annotation_df, name='new_annotation'):
    """Assign an annotation to the region

    Args:
        obj (AnnData/EMObject): region object
        annotation_df (pd.DataFrame): annotation dataframe, should contain only one column
        name (str): name of the annotation
    """
    assert sorted(annotation_df.index) == sorted(get_cell_ids(obj))
    assert annotation_df.shape[1] == 1
    annotation_df = annotation_df.loc[get_cell_ids(obj)]

    name = annotation_df.columns[0] if name is None else name
    if type(obj).__name__ == 'AnnData':
        obj.obs[name] = annotation_df[annotation_df.columns[0]]
    elif type(obj).__name__ == 'EMObject':
        obj.set_layer('raw')
        if name in obj.obs.columns:
            obj.del_anno(attr='obs', name=name, layer='raw')
        obj.add_anno(
            attr='obs',
            value=list(annotation_df[annotation_df.columns[0]]),
            name=name,
            layer='raw')
    else:
        warnings.warn("Unknown input format %s, skipping" % str(type(obj)))
    return


def assign_annotation_dict_to_objects(annotation_dict, objs, name='new_annotation'):
    """Assign an annotation dictionary to regions

    Args:
        annotation_dict (dict): annotation dictionary formatted as
            {(region id, cell id): annotation}
        objs (list): list of region objects
        name (str): annotation name
    """
    objs = [objs] if not isinstance(objs, list) else objs
    for obj in objs:
        region_id = get_name(obj)
        cell_ids = get_cell_ids(obj)
        annotation_as_list = [[annotation_dict[(region_id, cell_id)]] for cell_id in cell_ids]
        annotation_df = pd.DataFrame(np.array(annotation_as_list), index=cell_ids, columns=[name])
        assign_annotation(obj, annotation_df, name=name)
    return


def extract_feature_neighborhood_with_region_cell_ids(obj):
    """Extract the feature dataframe and spatial neighborhood dataframe from a
    region object, rename the index to include both region id and cell ids

    Args:
        obj (AnnData/EMObject): region object

    Returns:
        pd.Dataframe: feature dataframe
        pd.Dataframe: spatial neighborhood dataframe
    """
    region_name = get_name(obj)

    feature_df = deepcopy(get_feature(obj))
    feature_df['region_id'] = [region_name] * feature_df.shape[0]
    feature_df.rename_axis(index='CELL_ID', inplace=True)
    feature_df.reset_index(inplace=True)
    feature_df.set_index(['region_id', 'CELL_ID'], inplace=True)

    try:
        spatial_neighbor_df = deepcopy(get_cell_neighborhood(obj, neighbor_type='spatial'))
        spatial_neighbor_df.columns = ['spatial']
        spatial_neighbor_df['region_id'] = [region_name] * spatial_neighbor_df.shape[0]
        spatial_neighbor_df.rename_axis(index='CELL_ID', inplace=True)
        spatial_neighbor_df.reset_index(inplace=True)
        spatial_neighbor_df.set_index(['region_id', 'CELL_ID'], inplace=True)
        # Attach region id to cell ids
        spatial_neighbor_df['spatial'] = [[(region_name, n) for n in ns] for ns in spatial_neighbor_df['spatial']]
    except AssertionError:
        spatial_neighbor_df = None

    return feature_df, spatial_neighbor_df
