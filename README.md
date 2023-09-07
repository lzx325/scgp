# Spatial Cellular Graph Partition (SCGP)

This is an example code base for manuscript "Characterizing Tissue Structures From Spatial Omics with Spatial Cellular Graph Partition." DOI: [10.1101/2023.09.05.556133](https://doi.org/10.1101/2023.09.05.556133)

![SCGP_Image](https://gitlab.com/enable-medicine-public/scgp/uploads/a367f04dfd871d10d9f35e91312f9f24/Artboard_1gitlab_page.png)

## Installation
- from source:
    ```
    $ git clone https://gitlab.com/enable-medicine-public/scgp.git
    $ cd scgp
    $ python setup.py install
    ```

## Usage

See `Example.ipynb` for a tutorial of using SCGP and SCGP-Extension on regions from the DKD Kidney dataset

Use the `pytest` package to run the test suite:
```
$ pytest scgp/unit_tests.py
```

## Data Availability
Datasets used in the SCGP manuscript:
- DLPFC (Visium): [https://research.libd.org/spatialLIBD/](https://research.libd.org/spatialLIBD/)
- Healthy Lung IMC: [https://doi.org/10.5281/zenodo.6376766](https://doi.org/10.5281/zenodo.6376766)
- DKD Kidney: Raw data files stored under `data/DKD_Kidney`
- TR Kidney, UCSF Derm: currently being prepared and will be available soon. Stay tuned!

## Requirements

- numpy
- scipy
- pandas
- networkx
- scikit-learn
- matplotlib
- umap-learn
- igraph
- leidenalg
- emobject >= 0.7.3 or anndata
