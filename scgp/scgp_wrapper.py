import numpy as np
import pandas as pd
import time
import multiprocessing

from copy import deepcopy

from scgp.object_io import (
    get_name,
    get_cell_ids,
    get_biomarkers,
    get_cell_positions,
    get_cell_annotation,
    get_cell_neighborhood,
    get_normed_biomarker_expression,
    get_feature,
    extract_feature_neighborhood_with_region_cell_ids,
    assign_annotation_dict_to_objects,
)
from scgp.features import (
    calculate_feature,
    reweigh_features,
    get_unique_cell_types,
)
from scgp.neighborhood import (
    build_knn_neighborhood,
    build_distance_neighborhood,
    build_delaunay_triangulation_neighborhood,
    build_feature_knn_umap,
    construct_graph,
    count_edges_in_neighborhood,
)
from scgp.partition import (
    reorder_cluster_id,
    generic_clustering,
    leiden_partition,
    remove_isolated_patch,
    smooth_spatially_isolated_patch,
    membership_to_membership_dict,
)


def biomarker_clustering_wrapper(objs,
                                 seed=123,
                                 verbose=True,
                                 exclude_features=[],
                                 feature_weights={},
                                 method='Leiden',
                                 method_kwargs={'rp': 0.2, 'k': 20}):
    """Wrapper function for generic clustering of biomarker expression

    Args:
        objs (AnnData/EMObject/list): (list of) region object(s)
        seed (int, optional): random seed. Defaults to 123.
        verbose (bool, optional): verbosity. Defaults to True.
        exclude_features (list, optional): list of features to exclude.
        feature_weights (dict, optional): dict of feature weights.
            See `features.reweigh_features` for details.
        method (str, optional): clustering method. Defaults to 'Leiden'.
        method_kwargs (dict, optional): clustering kwargs. Defaults to {'rp': 0.2, 'k': 20}.

    Returns:
        dict: dict of cell(node) name: cell type cluster id
        tuple: tuple of features and model used for reproduction or finetuning
    """
    # Sanity check: all objects have the same list of biomarkers
    objs = [objs] if not isinstance(objs, list) else objs
    bms = get_biomarkers(objs[0])
    assert [get_biomarkers(obj) == bms for obj in objs]

    if seed is not None:
        np.random.seed(seed)

    # Collecting features: expression
    t0 = time.time()
    feature_item = {'expression': 1.}
    features = []
    for obj in objs:
        calculate_feature(obj, feature_item=feature_item)
        feature_df, _ = extract_feature_neighborhood_with_region_cell_ids(obj)
        feature_df = reweigh_features(
            feature_df,
            exclude_features=exclude_features,
            feature_weights=feature_weights)
        features.append(feature_df)
    features = pd.concat(features, axis=0)
    t_f = time.time()

    # Clustering
    bm_clusters, model = generic_clustering(features, method=method, method_kwargs=method_kwargs)
    t_c = time.time()
    if verbose:
        print("Featurization takes %.2fs, Clustering takes %.2fs" % (t_f - t0, t_c - t_f))
    return bm_clusters, (features, model)


def cellular_neighborhood_wrapper(objs,
                                  seed=123,
                                  verbose=True,
                                  exclude_features=[],
                                  feature_weights={},
                                  unique_cell_types=None,
                                  knn_neighbors=20,
                                  method='KMeans',
                                  method_kwargs={'n_clusters': 6}):
    """Wrapper function for generating cellular neighborhood clusters

    Args:
        objs (AnnData/EMObject/list): (list of) region object(s)
        seed (int, optional): random seed. Defaults to 123.
        verbose (bool, optional): verbosity. Defaults to True.
        exclude_features (list, optional): list of features to exclude.
        feature_weights (dict, optional): dict of feature weights.
            See `features.reweigh_features` for details.
        unique_cell_types (list, optional): list of unique cell types. Defaults to None.
        knn_neighbors (int, optional): number of closest spatial neighbors for
            calculating local composition vectors. Defaults to 20.
        method (str, optional): clustering method. Defaults to 'KMeans'.
        method_kwargs (dict, optional): clustering kwargs. Defaults to {'n_clusters': 6}.

    Returns:
        dict: dict of cell(node) name: cellular neighborhood cluster id
        tuple: tuple of features and model used for reproduction or finetuning
    """
    objs = [objs] if not isinstance(objs, list) else objs
    # Get a list of unique cell types
    _cell_types = set()
    for obj in objs:
        cts = get_unique_cell_types(obj)
        _cell_types.update(set(cts))
        if unique_cell_types is not None:
            assert len(set(cts) - set(unique_cell_types)) == 0
    unique_cell_types = unique_cell_types if unique_cell_types is not None else sorted(_cell_types)

    if seed is not None:
        np.random.seed(seed)

    # Collecting features (composition over kNN) for clustering
    t0 = time.time()
    feature_item = {'composition': 1.}
    features = []
    for obj in objs:
        build_knn_neighborhood(obj, k=knn_neighbors, attach_to_object=True)
        calculate_feature(obj, feature_item=feature_item, unique_cell_types=unique_cell_types)
        feature_df, _ = extract_feature_neighborhood_with_region_cell_ids(obj)
        feature_df = reweigh_features(
            feature_df,
            exclude_features=exclude_features,
            feature_weights=feature_weights)
        features.append(feature_df)
    features = pd.concat(features, axis=0)
    t_f = time.time()

    # Clustering
    cell_neighs, model = generic_clustering(features, method=method, method_kwargs=method_kwargs)
    t_c = time.time()
    if verbose:
        print("Featurization takes %.2fs, Clustering takes %.2fs" % (t_f - t0, t_c - t_f))
    return cell_neighs, (features, model)


def UTAG_wrapper(objs,
                 seed=122,
                 verbose=True,
                 exclude_features=[],
                 feature_weights={},
                 neighbor_distance_cutoff=20,
                 pixel_resolution=0.3775,
                 method='Leiden',
                 method_kwargs={'rp': 0.2, 'k': 20}):
    """Wrapper function for generating UTAG clusters

    Args:
        objs (AnnData/EMObject/list): (list of) region object(s)
        seed (int, optional): random seed. Defaults to 123.
        verbose (bool, optional): verbosity. Defaults to True.
        exclude_features (list, optional): list of features to exclude.
        feature_weights (dict, optional): dict of feature weights.
            See `features.reweigh_features` for details.
        neighbor_distance_cutoff (float, optional): distance cutoff (in um) for
            defining spatial neighborhood. Defaults to 20.
        pixel_resolution (float, optional): length (in um) per pixel.
        method (str, optional): clustering method. Defaults to 'KMeans'.
        method_kwargs (dict, optional): clustering kwargs. Defaults to {'n_clusters': 6}.

    Returns:
        dict: dict of cell(node) name: UTAG cluster id
        tuple: tuple of features and model used for reproduction or finetuning
    """
    # Sanity check: all objects have the same list of biomarkers
    objs = [objs] if not isinstance(objs, list) else objs
    bms = get_biomarkers(objs[0])
    assert [get_biomarkers(obj) == bms for obj in objs]

    if seed is not None:
        np.random.seed(seed)

    # Collecting features (smoothed_expression) for clustering
    t0 = time.time()
    feature_item = {'smoothed_expression': 1.}
    features = []
    for obj in objs:
        build_distance_neighborhood(
            obj,
            r=neighbor_distance_cutoff,
            um_per_px=pixel_resolution,
            attach_to_object=True)
        calculate_feature(obj, feature_item=feature_item)
        feature_df, _ = extract_feature_neighborhood_with_region_cell_ids(obj)
        feature_df = reweigh_features(
            feature_df,
            exclude_features=exclude_features,
            feature_weights=feature_weights)
        features.append(feature_df)
    features = pd.concat(features, axis=0)
    t_f = time.time()

    # Clustering
    utags, model = generic_clustering(features, method=method, method_kwargs=method_kwargs)
    t_c = time.time()
    if verbose:
        print("Featurization takes %.2fs, Clustering takes %.2fs" % (t_f - t0, t_c - t_f))
    return utags, (features, model)


def SLDA_wrapper(objs,
                 seed=123,
                 verbose=True,
                 exclude_features=[],
                 feature_weights={},
                 neighbor_distance_cutoff=20,
                 pixel_resolution=0.3775,
                 unique_cell_types=None,
                 n_topics=6,
                 difference_penalty=0.1,
                 n_parallel_processes=multiprocessing.cpu_count(),
                 admm_rho=0.1,
                 primal_dual_mu=2):
    """Wrapper function for generating Spatial LDA clusters

    Args:
        objs (AnnData/EMObject/list): (list of) region object(s)
        seed (int, optional): random seed. Defaults to 123.
        verbose (bool, optional): verbosity. Defaults to True.
        exclude_features (list, optional): list of features to exclude.
        feature_weights (dict, optional): dict of feature weights.
            See `features.reweigh_features` for details.
        neighbor_distance_cutoff (float, optional): distance cutoff (in um) for
            defining spatial neighborhood. Defaults to 20.
        pixel_resolution (float, optional): length (in um) per pixel.
        unique_cell_types (list, optional): list of unique cell types. Defaults to None.
        n_topics (int, optional): Number of LDA topics. Defaults to 6.
        difference_penalty (float, optional): prior for similarity between neighboring cells. Defaults to 0.1.
        n_parallel_processes (int, optional): Number of parallel processes.
        admm_rho (float, optional): optimization parameters. Defaults to 0.1.
        primal_dual_mu (float, optional): optimization parameters. Defaults to 2.

    Returns:
        dict: dict of cell(node) name: Spatial LDA topic id
        tuple: tuple of features and model used for reproduction or finetuning
    """
    objs = [objs] if not isinstance(objs, list) else objs
    # Get a list of unique cell types
    _cell_types = set()
    for obj in objs:
        cts = get_unique_cell_types(obj)
        _cell_types.update(set(cts))
        if unique_cell_types is not None:
            assert len(set(cts) - set(unique_cell_types)) == 0
    unique_cell_types = unique_cell_types if unique_cell_types is not None else sorted(_cell_types)

    if seed is not None:
        np.random.seed(seed)

    # Collecting features (composition over 20NN) for clustering
    t0 = time.time()
    feature_item = {'count': 1.}
    slda_features = []
    for obj in objs:
        build_distance_neighborhood(obj, r=neighbor_distance_cutoff, um_per_px=pixel_resolution, attach_to_object=True)
        calculate_feature(obj, feature_item=feature_item, unique_cell_types=unique_cell_types)
        feature_df, _ = extract_feature_neighborhood_with_region_cell_ids(obj)
        feature_df = reweigh_features(
            feature_df,
            exclude_features=exclude_features,
            feature_weights=feature_weights)
        slda_features.append(feature_df)
    slda_features = pd.concat(slda_features, axis=0)
    t_f = time.time()

    # Lazy import
    try:
        from spatial_lda.featurization import make_merged_difference_matrices
        from spatial_lda.model import train
    except Exception:
        print("Please install spatial lda from https://github.com/calico/spatial_lda")
        raise
    coord_dfs = {get_name(obj): get_cell_positions(obj) for obj in objs}
    diff_matrices = make_merged_difference_matrices(
        slda_features, coord_dfs, 'X', 'Y', z_col=None, reduce_to_mst=True)

    model = train(sample_features=slda_features,
                  difference_matrices=diff_matrices,
                  n_topics=n_topics,
                  difference_penalty=difference_penalty,
                  n_parallel_processes=n_parallel_processes,
                  admm_rho=admm_rho,
                  primal_dual_mu=primal_dual_mu)

    slda_topics = {r[0]: np.argmax(r[1]) for r in model.topic_weights.iterrows()}
    slda_topics = reorder_cluster_id(slda_topics)
    t_c = time.time()
    if verbose:
        print("Featurization takes %.2fs, Clustering takes %.2fs" % (t_f - t0, t_c - t_f))
    return slda_topics, ((slda_features, diff_matrices), model)


def SpaGCN_multi_regions_wrapper(objs,
                                 exclude_features=[],
                                 feature_weights={},
                                 seed=123,
                                 verbose=True,
                                 init='kmeans',
                                 n_clusters=7,
                                 n_neighbors=10,
                                 res=0.4):
    """Wrapper function for generating SpaGCN clusters on a list of regions

    Args:
        objs (AnnData/EMObject/list): (list of) region object(s)
        exclude_features (list, optional): list of features to exclude.
        feature_weights (dict, optional): dict of feature weights.
            See `features.reweigh_features` for details.
        seed (int, optional): random seed. Defaults to 123.
        verbose (bool, optional): verbosity. Defaults to True.
        init (str, optional): 'louvain' or 'kmeans'
        n_clusters (int, optional): number of clusters (for 'kmeans' init)
        n_neighbors (int, optional): number of neighbors (for 'lovain' init)
        res (float, optional): resolution (for 'louvain' init)

    Returns:
        dict: dict of cell(node) name: SpaGCN cluster id
        tuple: tuple of features and model used for reproduction or finetuning
    """
    import anndata as ad
    try:
        import SpaGCN as spg
    except Exception:
        print("Please install SpaGCN from https://github.com/jianhuupenn/SpaGCN")
        raise

    if seed is not None:
        np.random.seed(seed)

    # Build list of AnnData for SpaGCN
    t0 = time.time()
    cell_id_list = []
    adata_list = []
    adj_list = []
    l_list = []
    for obj in objs:
        calculate_feature(obj, feature_item={'expression': 1})
        feature_df = reweigh_features(
            get_feature(obj),
            exclude_features=exclude_features,
            feature_weights=feature_weights)
        adata = ad.AnnData(feature_df)
        cell_id_list.append(get_cell_ids(obj))
        adata.var_names = get_biomarkers(obj)
        coords = get_cell_positions(obj)
        adata.obs["X"] = coords["X"]
        adata.obs["Y"] = coords["Y"]
        adata.uns["name"] = get_name(obj)
        adata_list.append(adata)
        adj = spg.calculate_adj_matrix(x=list(coords["X"]), y=list(coords["Y"]), histology=False)
        adj_list.append(adj)
        length_factor = spg.search_l(0.5, adj, start=0.01, end=1000, tol=0.01, max_run=100)
        l_list.append(length_factor)
    t_f = time.time()

    # Run multi-region SpaGCN
    clf = spg.multiSpaGCN()
    clf.train(
        adata_list, adj_list, l_list, num_pcs=20, lr=0.005, weight_decay=0.001,
        opt="admin", init=init, n_clusters=n_clusters, n_neighbors=n_neighbors,
        res=res, tol=1e-3)
    y_pred, _ = clf.predict()

    # Spatial smoothing of clusters
    cur = 0
    refined_pred = []
    for cids, adj in zip(cell_id_list, adj_list):
        num_cells = len(cids)
        assert adj.shape == (num_cells, num_cells)
        _pred = y_pred[cur:(cur + num_cells)]
        _refined_pred = spg.refine(
            sample_id=np.arange(len(_pred)), pred=list(_pred), dis=adj, shape="hexagon")
        refined_pred.extend(_refined_pred)
        cur += num_cells
    assert cur == len(y_pred)

    keys = []
    for obj, cids in zip(objs, cell_id_list):
        keys.extend([(get_name(obj), cid) for cid in cids])
    refined_results = dict(zip(keys, refined_pred))

    t_c = time.time()
    if verbose:
        print("Featurization takes %.2fs, Clustering takes %.2fs" % (t_f - t0, t_c - t_f))
    return refined_results, ((adata_list, adj_list, l_list), clf)


def SpaGCN_wrapper(obj, res=0.4):
    """Wrapper function for generating SpaGCN clusters on a single region

    Args:
        obj (AnnData/EMObject): region object
        res (float, optional): resolution

    Returns:
        dict: dict of cell(node) name: SpaGCN cluster id
        tuple: tuple of features and model used for reproduction or finetuning
    """
    import anndata as ad
    try:
        import SpaGCN as spg
    except Exception:
        print("Please install SpaGCN from https://github.com/jianhuupenn/SpaGCN")
        raise

    adata = ad.AnnData(get_normed_biomarker_expression(obj))
    adata.var_names = get_biomarkers(obj)
    adata.uns["name"] = get_name(obj)

    coords = get_cell_positions(obj)
    x, y = list(coords['X']), list(coords['Y'])

    adj = spg.calculate_adj_matrix(x=x, y=y, histology=False)
    length_factor = spg.search_l(0.5, adj, start=0.01, end=1000, tol=0.01, max_run=100)

    clf = spg.SpaGCN()
    clf.set_l(length_factor)
    clf.train(adata, adj, num_pcs=20, res=res, tol=5e-3, lr=0.05, max_epochs=200)
    y_pred, _ = clf.predict()
    refined_pred = spg.refine(sample_id=np.arange(adata.shape[0]), pred=list(y_pred), dis=adj, shape="hexagon")

    refined_pred_dict = {(get_name(obj), cid): p for cid, p in zip(get_cell_ids(obj), refined_pred)}
    return refined_pred_dict, ((adata, adj, length_factor), clf)


def SCGP_wrapper(objs,
                 exclude_features=[],
                 feature_weights={},
                 seed=122,
                 verbose=True,
                 delaunay_distance_cutoff=35,
                 pixel_resolution=0.3775,
                 rp=1e-3,
                 feature_knn=5,
                 smooth_level=0,
                 smooth_iter=1,
                 attach_to_object=True):
    """Wrapper function for SCGP

    Args:
        objs (AnnData/EMObject/list): (list of) region object(s)
        exclude_features (list, optional): list of features to exclude.
        feature_weights (dict, optional): dict of feature weights.
            See `features.reweigh_features` for details.
        seed (int, optional): random seed. Defaults to 123.
        verbose (bool, optional): verbosity. Defaults to True.
        delaunay_distance_cutoff (float, optional): distance cutoff (in um) for
            excluding distant delaunay triangulation edges. Defaults to 35.
        pixel_resolution (float, optional): length (in um) per pixel.
        rp (float, optional): resolution parameter for leiden partition.
        feature_knn (int, optional): number of nearest neighbors in the
            feature space
        smooth_level (int, optional): smooth level for post partition
            smoothing, see `scgp.partition.smooth_spatially_isolated_patch`
        smooth_iter (int, optional): number of smoothing runs
        attach_to_object (bool, optional): if to attach the resulting SCGP
            partitions to the input region object(s).

    Returns:
        dict: dict of cell(node) name: SCGP partition id
        tuple: tuple of features and model used for reproduction or finetuning
    """
    # Sanity check: all objects have the same list of biomarkers
    objs = [objs] if not isinstance(objs, list) else objs
    bms = get_biomarkers(objs[0])
    assert [get_biomarkers(obj) == bms for obj in objs]

    if seed is not None:
        np.random.seed(seed)

    # Collecting features (smoothed_expression) for clustering
    t0 = time.time()
    feature_item = {'expression': 1}
    features = []
    spatial_neighbors_df = []
    for obj in objs:
        build_delaunay_triangulation_neighborhood(
            obj, r=delaunay_distance_cutoff, um_per_px=pixel_resolution, attach_to_object=True)
        calculate_feature(obj, feature_item=feature_item)
        # Extract feature and neighborhood, with rows indexed by (region_id, cell_id)
        feature_df, neighbor_df = extract_feature_neighborhood_with_region_cell_ids(obj)
        feature_df = reweigh_features(
            feature_df,
            exclude_features=exclude_features,
            feature_weights=feature_weights)
        features.append(feature_df)
        spatial_neighbors_df.append(neighbor_df)

    features = pd.concat(features, axis=0)
    spatial_neighbors_df = pd.concat(spatial_neighbors_df, axis=0)
    feature_neighbors_df = build_feature_knn_umap(features, k=feature_knn, seed=seed, attach_to_object=False)
    all_neighbors_df = pd.concat([spatial_neighbors_df, feature_neighbors_df], axis=1)

    if verbose:
        n_spatial_edges = count_edges_in_neighborhood(all_neighbors_df[['spatial']])
        n_feature_edges = count_edges_in_neighborhood(all_neighbors_df[['feature']])
        print("Use %d spatial edges and %d feature edges" % (n_spatial_edges, n_feature_edges))

    t_f = time.time()
    nx_graph = construct_graph(
        all_neighbors_df, feature_df=features, weighted=True, normalize=True)
    scgp_membership = SCGP_partition(
        (features, all_neighbors_df), nx_graph, rp=rp, smooth_level=smooth_level,
        smooth_iter=smooth_iter, verbose=verbose)
    t_c = time.time()
    if verbose:
        print("Featurization takes %.2fs, Clustering takes %.2fs" % (t_f - t0, t_c - t_f))

    if attach_to_object:
        assign_annotation_dict_to_objects(
            scgp_membership, objs, name='SCGP_annotation', categorical=True)
    return scgp_membership, ((features, all_neighbors_df), nx_graph)


def SCGP_partition(feats,
                   nx_graph,
                   rp=1e-3,
                   smooth_level=0,
                   smooth_iter=1,
                   verbose=False):
    """Partitioning hybrid graphs defined by SCGP

    Args:
        feats (tuple): tuple of feature and neighborhood data frames.
        nx_graph (nx.Graph): hybrid graph representatino of region(s).
        rp (float, optional): resolution parameter for leiden partition.
        smooth_level (int, optional): smooth level for post partition
            smoothing, see `scgp.partition.smooth_spatially_isolated_patch`.
        smooth_iter (int, optional): number of smoothing runs.

    Returns:
        dict: dict of cell(node) name: SCGP partition id
    """
    features, all_neighbors_df = feats
    membership = leiden_partition(nx_graph, rp=rp)
    membership = remove_isolated_patch(membership)
    for _ in range(smooth_iter):
        membership = smooth_spatially_isolated_patch(
            all_neighbors_df[['spatial']], membership, smooth_level=smooth_level)
    membership = remove_isolated_patch(membership)
    if verbose:
        print("Find %d partitions" % (len(set(membership)) - 1))  # Remove unknown

    # Define cell type dict
    scgp_membership = membership_to_membership_dict(membership, features)
    return scgp_membership
