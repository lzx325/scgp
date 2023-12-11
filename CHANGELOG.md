# Changelog

### 0.1.0

- Initial version of SCGP (Spatial Cellular Graph Partition)

### 0.2.0

- Add support for squidpy graph-related functions
    - In `anndata` mode, coordinates will be saved as `spatial` entry under `obsm`
    - In `anndata` mode, neighborhood will be saved as csr matrices as `X_connectivities` entries under `obsp`
- `SCGP_wrapper` now automatically attaches partitions to the region object(s)