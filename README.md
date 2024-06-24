# Spatial Cellular Graph Partition (SCGP)

This is the code base for article "Characterizing Tissue Structures From Spatial Omics with Spatial Cellular Graph Partition." DOI: [10.1101/2023.09.05.556133](https://doi.org/10.1101/2023.09.05.556133)

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
- [DLPFC (Visium)](https://research.libd.org/spatialLIBD/)
- [Healthy Lung (IMC)](https://doi.org/10.5281/zenodo.6376766)
- [Adult Mouse Brain (MERFISH)](https://alleninstitute.github.io/abc_atlas_access/notebooks/zhuang_merfish_tutorial.html)
- [UPMC-HNC (CODEX)](https://app.enablemedicine.com/portal/atlas-library/studies/92394a9f-6b48-4897-87de-999614952d94)
- DKD Kidney, TR Kidney, UCSF Derm, Stanford-PC (CODEX): raw files are currently being assembled and will be available soon. Stay tuned!

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
- anndata or emobject >= 0.7.3
