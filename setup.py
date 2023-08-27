from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')  # noqa

# get version
exec(open('scgp/version.py').read())


setup(
    name='SCGP',
    version=__version__,  # noqa
    description='Code base for Spatial Cellular Graph Partition (SCGP)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://gitlab.com/enable-medicine-public/scgp',
    author="Zhenqin Wu",
    author_email='zhenqin@enablemedicine.com;zqwu@stanford.edu',
    packages=find_packages(),
    python_requires='>=3.7, <4',
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "networkx",
        "matplotlib",
        "scikit-learn",
        "umap-learn",
        "igraph",
        "leidenalg",
        "emobject>=0.7.3",
    ],
)
