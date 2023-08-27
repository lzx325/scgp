import matplotlib
import matplotlib.pyplot as plt


# def plot_all_acquisitions_with_annotations(pred_dict, objs, path_prefix=None, file_format='png', figsize=10, **kwargs):
#     """Plot (a list of) region(s) colored by a dict of membership/cluster

#     Args:
#         pred_dict (dict): dict of cell(node) name: cluster id
#         objs (AnnData/EMObject/list): (list of) region object(s)
#         path_prefix (str, optional): path prefix for saving figures. Use `None` for inline plotting
#         figsize (int, optional): figure size. Defaults to 10.
#     """
#     objs = [objs] if not isinstance(objs, list) else objs
#     if path_prefix is not None:
#         print("Plotting %d graphs to %s" % (len(objs), str(path_prefix)))
#     for obj in objs:
#         region_id = get_name(obj)
#         node_gp = [pred_dict[(region_id, cell_id)] for cell_id in get_cell_ids(obj)]
#         node_colors = [matplotlib.cm.tab20(i % 20) for i in node_gp]
#         plt.clf()
#         plt.figure(figsize=(figsize, figsize))
#         plot_acquisition(obj, node_colors=node_colors, **kwargs)
#         plt.gca().invert_yaxis()
#         plt.axis('off')
#         if path_prefix is None:
#             plt.show()
#         else:
#             if file_format == 'png':
#                 plt.savefig('%s-%s.png' % (path_prefix, region_id), dpi=300, transparent=True)
#             elif file_format == 'pdf':
#                 plt.savefig('%s-%s.pdf' % (path_prefix, region_id), transparent=True)
#             else:
#                 raise ValueError
#     return


# def plot_acquisition(obj,
#                      node_colors=None,
#                      node_plotting_order=None,
#                      plot_edges=False,
#                      subsample_edges=True,
#                      edge_prop={},
#                      node_prop={}):
#     """ Plot a region as a scatter plot

#     Args:
#         obj (AnnData/EMObject): region object
#         node_colors (list/None, optional): list of node colors
#         plot_edges (bool, optional): if to plot spatial edges
#         subsample_edges (bool, optional): if to subsample edges for plotting
#         edge_props (dict, optional): additional kwargs for plotting edges
#         node_props (dict, optional): additional kwargs for plotting nodes

#     """
#     cell_ids = get_cell_ids(obj)
#     node_coords = get_cell_positions(obj)

#     node_colors = ['b'] * len(cell_ids) \
#         if node_colors is None else list(node_colors)
#     node_plotting_order = [0] * len(cell_ids) \
#         if node_plotting_order is None else list(node_plotting_order)
#     assert len(node_colors) == node_coords.shape[0]
#     assert len(node_plotting_order) == node_coords.shape[0]

#     if plot_edges:
#         neighbor_df = get_cell_neighborhood(obj, neighbor_type='spatial')
#         edges = []
#         for cell_id, ns in neighbor_df.sum(1).iteritems():
#             xi, yi = node_coords.loc[cell_id, 'X'], node_coords.loc[cell_id, 'Y']
#             for n in ns:
#                 if n != cell_id:
#                     xj, yj = node_coords.loc[n, 'X'], node_coords.loc[n, 'Y']
#                     edges.append((xi, yi, xj, yj))
#         np.random.shuffle(edges)
#         if subsample_edges and len(edges) > len(cell_ids) * 6:
#             # Sub-sample edges to speed up plotting
#             edges = edges[:(6 * len(cell_ids))]
#         edge_kwargs = {'c': (0.4, 0.4, 0.4, 1.0), 'linewidth': 0.5}
#         edge_kwargs.update(edge_prop)
#         for e in edges:
#             plt.plot([e[0], e[2]],
#                      [e[1], e[3]],
#                      zorder=1,
#                      **edge_kwargs)

#     node_kwargs = {'s': 8}
#     node_kwargs.update(node_prop)
#     for group in sorted(set(node_plotting_order)):
#         group_inds = np.where(np.array(node_plotting_order) == group)[0]
#         plt.scatter(list(node_coords['X'].iloc[group_inds]),
#                     list(node_coords['Y'].iloc[group_inds]),
#                     c=[node_colors[i] for i in group_inds],
#                     zorder=2,
#                     **node_kwargs)
#     plt.gca().set_aspect('equal')
#     return


# def plot_heatmap(pred_dict, objs, use_clusters=[], use_features=[], feature_axis_names=[], path_prefix=None):
#     """Plot heatmap on the feature for each cluster

#     Args:
#         pred_dict (dict): dict of cell(node) name: cluster id
#         objs (AnnData/EMObject/list): (list of) region object(s)
#         path_prefix (str, optional): path prefix for saving figures. Use `None` for inline plotting

#     Returns:
#         np.ndarray: enrichment matrix for features
#     """
#     objs = [objs] if not isinstance(objs, list) else objs
#     for obj in objs:
#         try:
#             get_feature(obj)
#         except Exception:
#             calculate_feature(obj, feature_item={'expression': 1.})
#     if path_prefix is not None:
#         print("Plotting heatmap to %s" % str(path_prefix))

#     # Specify rows
#     valid_clusters = set()
#     for cluster_id, count in zip(*np.unique(list(pred_dict.values()), return_counts=True)):
#         # Only plotting compartments that contain enough samples
#         if count > 0.005 * len(pred_dict) and cluster_id >= 0:
#             valid_clusters.add(cluster_id)

#     use_clusters = sorted(valid_clusters) if len(use_clusters) == 0 else use_clusters

#     feature_by_clusters = {cl_id: [] for cl_id in use_clusters}
#     for obj in objs:
#         region_id = get_name(obj)
#         cluster_ids = [pred_dict[(region_id, cell_id)] for cell_id in get_cell_ids(obj)]
#         feature_df = get_feature(obj)
#         for c in use_clusters:
#             feature_by_clusters[c].append(np.array(feature_df.iloc[np.where(np.array(cluster_ids) == c)]))

#     # Specify columns
#     mean_heatmap = [np.concatenate(feature_by_clusters[cl_id], 0).mean(0) for cl_id in use_clusters]
#     mean_heatmap = np.stack(mean_heatmap, 0)

#     feature_names = list(get_feature(objs[0]).columns)
#     use_features = sorted(feature_names) if len(use_features) == 0 else use_features
#     use_cols = [feature_names.index(f) for f in use_features]
#     mean_heatmap = mean_heatmap[:, use_cols]

#     feature_axis_names = use_features if len(feature_axis_names) == 0 else feature_axis_names
#     assert len(feature_axis_names) == len(use_features)

#     plt.clf()
#     plt.figure(figsize=(10, 3))
#     plt.imshow(mean_heatmap, vmin=-1.5, vmax=1.5, cmap='bwr')
#     plt.xticks(np.arange(len(use_features)), feature_axis_names, rotation=90)
#     plt.yticks([])
#     if path_prefix is None:
#         plt.show()
#     else:
#         plt.savefig('%s-expression_by_compartment.png' % path_prefix, dpi=300, bbox_inches='tight')

#     plt.clf()
#     plt.figure(figsize=(1, 3))
#     plt.imshow(np.array(use_clusters).reshape((-1, 1)), cmap='tab20', vmin=0, vmax=19)
#     plt.axis('off')
#     if path_prefix is None:
#         plt.show()
#     else:
#         plt.savefig('%s-expression_by_compartment_legend.png' % path_prefix, dpi=300, bbox_inches='tight')
#     return mean_heatmap