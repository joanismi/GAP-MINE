"""APID Database Download already allows the inclusion of HGNC Unique
Identifiers in the data. However, some of these IDs are
deprecated and were replaced by newer versions, so each ID should be
confirmed to be in its most recent version."""

import pandas as pd
import id_conversion

apid = pd.read_csv(
    "python/data/raw/9606_noISI_Q2.txt",
      sep='\t', header=0, index_col='InteractionID'
      )

hgnc_symbols = pd.read_csv(
    "python/data/raw/hgnc_reference_table.txt",
      sep='\t', header=0
      )

genenames = set(list(apid['GeneName_A'].unique()) + list(apid['GeneName_B'].unique()))  # set of unique apid genes

hgnc_ids = list(hgnc_symbols['Approved symbol'].unique())# apid ids not found in hgnc table
missing_apid_ids = [protein for protein in genenames if protein not in hgnc_ids]

confirmed_hgnc, ids_not_found = id_conversion.confirm_hgnc(missing_apid_ids)

for hgnc_id, old_ids in confirmed_hgnc.items():
    for old_id in old_ids:
        if old_id != None:
            apid.loc[apid['GeneName_A'] == old_id, 'GeneName_A'] = hgnc_id
            apid.loc[apid['GeneName_B'] == old_id, 'GeneName_B'] = hgnc_id

missing_apid_ids_dict = {'HGNC ID': [], 'Approved symbol': [], 'NCBI Gene ID(supplied by NCBI)': [], 'Ensembl ID(supplied by Ensembl)': [], 'UniProt ID(supplied by UniProt)': []}
for id in ids_not_found:
    missing_apid_ids_dict['HGNC ID'].append(None)
    missing_apid_ids_dict['Approved symbol'].append(id)
    missing_apid_ids_dict['NCBI Gene ID(supplied by NCBI)'].append(None)
    missing_apid_ids_dict['Ensembl ID(supplied by Ensembl)'].append(None)
    if len(apid[apid['GeneName_A'] == id]['UniprotID_A'].values) == 0:
        uniprot_id = apid[apid['GeneName_B'] == id]['UniprotID_B'].values[0]
    else:
        uniprot_id = apid[apid['GeneName_A'] == id]['UniprotID_A'].values[0]
    missing_apid_ids_dict['UniProt ID(supplied by UniProt)'].append(uniprot_id)
missing_apid_ids_df = pd.DataFrame.from_dict(missing_apid_ids_dict)

hgnc_symbols = pd.concat([hgnc_symbols, missing_apid_ids_df])
hgnc_symbols.to_csv('python/data/interim/hgnc_symbols.csv', header=-1, index=False)

apid = apid[['GeneName_A', 'GeneName_B']]
apid.to_csv('python/data/interim/apid.csv', header=-1, index=False)
