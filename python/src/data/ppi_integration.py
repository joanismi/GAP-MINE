import pandas as pd
import numpy as np

apid = pd.read_csv("././data/interim/apid.csv", sep=',',
                   header=0, names=['HGNC_A', 'HGNC_B'])
apid['apid'] = True
apid.dropna(inplace=True)

huri = pd.read_csv("././data/interim/HuriHGNC_IDs.csv", sep=',', header=0)
huri['huri'] = True

ppi_interactions = pd.merge(apid[['HGNC_A', 'HGNC_B', 'apid']], huri[['HGNC_A', 'HGNC_B', 'huri']], on=[
                            'HGNC_A', 'HGNC_B'], how='outer')
ppi_interactions.fillna(False, inplace=True)

ppi_interactions.to_csv(
    '././data/processed/ppis/apid_huri_ppis.csv', index=False, header=-1)

disgenet = pd.read_csv(
    "././data/raw/curated_gene_disease_associations.tsv", sep='\t', header=0)
disgenet.to_csv('././data/interim/disgenet.csv', header=-1, index=False)
