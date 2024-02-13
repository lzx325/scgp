import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import Delaunay, KDTree

from scgp.object_io import (
    get_cell_ids,
    get_cell_positions,
    get_feature,
    assign_neighborhood,
)


def build_distance_neighborhood(coords, r=20, um_per_px=0.3775, attach_to_object=False):
    """Find neighbors within a distance threshold

    Args:
        coords (np.ndarray/pd.DataFrame/EMObject/AnnData): cell coordinates
        r (float): distance threshold (in um)
        um_per_px (float): size (in um) of pixel
        attach_to_object (bool): when input is an EMObject or AnnData,
            whether to attach neighbor cell indices to the object

    Returns:
        neighbor_df (pd.DataFrame): dataframe of spatial neighbor list
    """
    if isinstance(coords, np.ndarray):
        coord_ar = coords
        cell_ids = np.arange(coord_ar.shape[0])
    elif isinstance(coords, pd.DataFrame):
        coord_ar = np.array(coords)
        cell_ids = coords.index
    else:
        coord_ar = np.array(get_cell_positions(coords))
        cell_ids = get_cell_ids(coords)

    r = r / um_per_px  # convert to px for internal calcs
    tree = KDTree(coord_ar)
    neighbors = tree.query_ball_point(coord_ar, r=r)
    neighbors = [list(n) for n in neighbors]  # index - index
    neighbors = [[cell_ids[_n] for _n in n] for n in neighbors]  # index - cell_id
    neighbor_df = pd.DataFrame({"neighbors": neighbors}, index=cell_ids)  # cell_id - cell_id

    if attach_to_object:
        assign_neighborhood(coords, neighbor_df, neighbor_type='spatial')
    return neighbor_df


def build_knn_neighborhood(coords, k=10, return_neighbor_distances=False, attach_to_object=False):
    """Find k nearest neighbors

    Args:
        coords (np.ndarray/pd.DataFrame/EMObject/AnnData): cell coordinates
        k (int) : number of nearest neighbors to find
        return_neighbor_distances (bool): if to return distances (in pixel)
            for the k nearest neighbors
        attach_to_object (bool): when input is an EMObject or AnnData,
            whether to attach neighbor cell indices to the object

    Returns:
        neighbor_df (pd.DataFrame): dataframe of spatial neighbor list
    """
    if isinstance(coords, np.ndarray):
        coord_ar = coords
        cell_ids = np.arange(coord_ar.shape[0])
    elif isinstance(coords, pd.DataFrame):
        coord_ar = np.array(coords)
        cell_ids = coords.index
    else:
        coord_ar = np.array(get_cell_positions(coords))
        cell_ids = get_cell_ids(coords)

    tree = KDTree(coord_ar)
    neighbor_distances, neighbors = tree.query(coord_ar, k=k)
    neighbors = [list(n) for n in neighbors]  # index - index
    neighbors = [[cell_ids[_n] for _n in n] for n in neighbors]  # index - cell_id
    neighbor_df = pd.DataFrame({"neighbors": neighbors}, index=cell_ids)  # cell_id - cell_id

    if attach_to_object:
        assign_neighborhood(coords, neighbor_df, neighbor_type='spatial')

    if return_neighbor_distances:
        return neighbor_df, neighbor_distances
    else:
        return neighbor_df


def build_delaunay_triangulation_neighborhood(coords, r=None, um_per_px=0.3775, attach_to_object=False):
    """Find neighbors using Delaunay triangulation

    Args:
        coords (np.ndarray/pd.DataFrame/EMObject/AnnData): cell coordinates
        r (float): distance threshold (in um). If not None, will prune
            neighborhood based on distance threshold
        um_per_px (float): size (in um) of pixel
        attach_to_object (bool): when input is an EMObject or AnnData,
            whether to attach neighbor cell indices to the object

    Returns:
        neighbor_df (pd.DataFrame): dataframe of spatial neighbor list
    """
    if isinstance(coords, np.ndarray):
        coord_ar = coords
        cell_ids = np.arange(coord_ar.shape[0])
    elif isinstance(coords, pd.DataFrame):
        coord_ar = np.array(coords)
        cell_ids = coords.index
    else:
        coord_ar = np.array(get_cell_positions(coords))
        cell_ids = get_cell_ids(coords)

    dln = Delaunay(coord_ar)
    neighbors = [set() for _ in range(len(coord_ar))]
    for t in dln.simplices:
        for v in t:
            neighbors[v].update(t)
    neighbors = [list(n) for n in neighbors]  # index - index
    if r is not None:
        neighbors = prune_graph_by_distance(neighbors, coord_ar, r=r, um_per_px=um_per_px)

    neighbors = [[cell_ids[_n] for _n in n] for n in neighbors]  # index - cell_id
    neighbor_df = pd.DataFrame({"neighbors": neighbors}, index=cell_ids)  # cell_id - cell_id
    if attach_to_object:
        assign_neighborhood(coords, neighbor_df, neighbor_type='spatial')
    return neighbor_df


def prune_graph_by_distance(neighbors, coord_arr, r=20, um_per_px=0.3775):
    """Prune neighborhood based on a distance threshold.

    For example, if distance threshold is 100px, and Deluanay triangulation
    yields edges between nodes that exceed the distance threshold. These
    edges will be removed from the neighborhood.

    Args:
        neighbors (list): list of neighbors
        coord_arr (np.ndarray): numpy array of cell coordinates
        r (float): distance threshold (in um)
        um_per_px (float): size (in um) of pixel

    Returns:
        neighbors (list): pruned list of neighbors
    """
    r = r / um_per_px  # convert to pixels

    # First calculate what the acceptable edges are given distance threshold
    tree = KDTree(coord_arr)
    _neighbors = tree.query_ball_point(coord_arr, r=r)
    neighbors = [list(set(n) & set(_n)) for n, _n in zip(neighbors, _neighbors)]
    return neighbors


def build_feature_knn_kdtree(features, k=10, attach_to_object=False):
    """Find k nearest neighbors in the feature space

    Args:
        features (np.ndarray/pd.DataFrame/EMObject/AnnData): cellular features
        k (int) : number of nearest neighbors to find
        attach_to_object (bool): when input is an EMObject or AnnData,
            whether to attach feature neighbor cell indices to the object

    Returns:
        feature_neighbor_df (pd.DataFrame): dataframe of feature neighbor list
    """
    if isinstance(features, np.ndarray):
        feature_df = pd.DataFrame(features, index=np.arange(features.shape[0]))
    elif isinstance(features, pd.DataFrame):
        feature_df = features
    else:
        feature_df = get_feature(features)  # This will return a pd.DataFrame

    feature_neighbor_df = build_knn_neighborhood(feature_df, k=k + 1)  # cell_id - cell_id
    for i in feature_neighbor_df.index:
        feature_neighbor_df.loc[i, 'neighbors'].remove(i)

    feature_neighbor_df.columns = ['feature']
    if attach_to_object:
        assign_neighborhood(features, feature_neighbor_df, neighbor_type='feature')
    return feature_neighbor_df


def build_feature_knn_umap(features, k=10, seed=123, attach_to_object=False):
    """Find k nearest neighbors in the feature space

    This method uses nearest neighbor method from umap-learn (pynndescent)

    Args:
        features (np.ndarray/pd.DataFrame/EMObject/AnnData): cellular features
        k (int) : number of nearest neighbors to find
        seed (int): random seed
        attach_to_object (bool): when input is an EMObject or AnnData,
            whether to attach feature neighbor cell indices to the object

    Returns:
        feature_neighbor_df (pd.DataFrame): dataframe of feature neighbor list
    """
    from umap.umap_ import nearest_neighbors
    from umap.umap_ import fuzzy_simplicial_set

    if isinstance(features, np.ndarray):
        feature_ar = features
        cell_ids = np.arange(feature_ar.shape[0])
    elif isinstance(features, pd.DataFrame):
        feature_ar = np.array(features)
        cell_ids = features.index
    else:
        feature_ar = np.array(get_feature(features))
        cell_ids = get_cell_ids(features)

    feature_neighbors = [[] for _ in range(feature_ar.shape[0])]
    if k == 0:
        feature_neighbor_df = pd.DataFrame({"feature": feature_neighbors}, index=cell_ids)
        return feature_neighbor_df
        
    knn_indices, knn_dists, forest = nearest_neighbors(
        X=feature_ar,
        n_neighbors=k,
        metric='euclidean',
        metric_kwds={},
        angular=False,
        random_state=np.random.RandomState(seed=seed),
        verbose=False,
    )

    connectivities = fuzzy_simplicial_set(
        feature_ar,
        k,
        None,
        None,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
    )

    adjacency = connectivities[0]
    sources, targets = adjacency.nonzero()

    assert sources.shape == targets.shape
    for source, target in zip(sources, targets):
        feature_neighbors[source].append(target)  # index - index

    feature_neighbors = [[cell_ids[_n] for _n in n] for n in feature_neighbors]  # index - cell_id
    feature_neighbor_df = pd.DataFrame({"feature": feature_neighbors}, index=cell_ids)  # cell_id - cell_id

    if attach_to_object:
        assign_neighborhood(features, feature_neighbor_df, neighbor_type='feature')
    return feature_neighbor_df


def count_edges_in_neighborhood(neighbor_df):
    """Count number of undirected edges (neighboring pairs of nodes)

    Args:
        neighbor_df (pd.DataFrame): dataframe of neighbor list

    Returns:
        int: number of undirected edges
    """
    edges = set()
    for cell_id, ns in neighbor_df.sum(1).items():
        for n in ns:
            if cell_id != n:
                edges.add((cell_id, n))
                edges.add((n, cell_id))
    return len(edges) / 2


def get_node_pair_similarity(n1, n2s, feature_df):
    """Calculate similarity between cell `n1` and a list of neighbor cells `n2s`

    Args:
        n1 (Any): cell id of the query cell
        n2s (Any/list): (list of) cell id(s) of the neighbor cell(s)
        feature_df (pd.DataFrame): feature dataframe

    Returns:
        float/np.ndarray: distance(s) between the query cell and its neighbor cell(s)
    """
    f1 = np.array(feature_df.loc[n1]).reshape((1, -1))

    if not isinstance(n2s, list) and n2s in feature_df.index:
        f2s = np.array(feature_df.loc[[n2s]])
        feature_diff = np.linalg.norm(f1 - f2s, ord=2, axis=1)
        similarity = 1 / np.clip(feature_diff, 1e-5, np.inf)
        return float(similarity)
    else:
        f2s = np.array(feature_df.loc[n2s])
        feature_diff = np.linalg.norm(f1 - f2s, ord=2, axis=1)
        similarity = 1 / np.clip(feature_diff, 1e-5, np.inf)
        return similarity


def construct_graph(neighbor_df, feature_df=None, weighted=False, normalize=False):
    """Construct a networkx graph based on neighbor list

    Args:
        neighbor_df (pd.DataFrame): dataframe of neighbor list
        feature_df (pd.DataFrame): feature dataframe
        weighted (bool): If to assign weights to edges,
            only applicable when `feature_df` is provided
        normalize (bool): If to normalize weights

    Returns:
        G (nx.Graph): neighborhood graph
    """
    all_neighbors = neighbor_df.sum(1)

    G = nx.Graph()
    for cell_id in all_neighbors.index:
        G.add_node(cell_id, cell_id=cell_id)
    if feature_df is not None:
        assert sorted(feature_df.index) == sorted(neighbor_df.index)
        for cell_id, feature in feature_df.iterrows():
            G.nodes[cell_id]['feature'] = feature

    all_edges = set()
    for cell_id, ns in zip(all_neighbors.index, all_neighbors):
        for n in ns:
            all_edges.add((min(cell_id, n), max(cell_id, n)))
    all_edges = list(all_edges)

    if weighted:
        f1 = feature_df.loc[[e[0] for e in all_edges]]
        f2 = feature_df.loc[[e[1] for e in all_edges]]
        feature_diff = np.linalg.norm(np.array(f1) - np.array(f2), axis=1, ord=2)
        similarity = 1 / np.clip(feature_diff, 1e-5, np.inf)
        weights = np.clip(similarity / np.median(similarity), 1e-5, 3)
    else:
        weights = [1] * len(all_edges)

    for e, w in zip(all_edges, weights):
        G.add_edge(e[0], e[1], weight=w)

    return G
