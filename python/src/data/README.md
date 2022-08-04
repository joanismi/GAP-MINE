About the files:

Despite dealing with different datasets, the conversion scripts have a dataset in common (with the conversion IDs) that can be updated during the execution of the scripts.
To ensure a smooth running of the scripts, please execute them in the following order:

1. apid2hgnc.py
2. ensembl2hgnc.py
3. omnipath2hgnc.py
4. ncbi2hgnc.py 
5. dorothea_load.py
6. ppi_integration.py

The file id_conversion.py contains the functions used in the above-mentioned scripts.