import pandas as pd
import id_conversion

pd.set_option('precision', 0)
pd.set_option('display.float_format', lambda x: '%.0f' % x)

hgnc_reference_table = pd.read_csv("././data/interim/HGNC symbols.txt", sep='\t', header=0)
hgnc_reference_table.drop_duplicates(inplace=True)
huri = pd.read_csv("././data/raw/HuRI.tsv", sep='\t',header=None, names=['ENSG A', 'ENSG B'])

huri_id = set(list(huri['ENSG A'].unique()) + list(huri['ENSG B'].unique()))
for id in huri_id:
    hgnc_id = hgnc_reference_table.loc[hgnc_reference_table['Ensembl ID(supplied by Ensembl)'] == id, 'Approved symbol'].values
    if len(hgnc_id) > 0:
        huri.loc[huri['ENSG A'] == id, 'HGNC_A'] = hgnc_id[0]
        huri.loc[huri['ENSG B'] == id, 'HGNC_B'] = hgnc_id[0]

huri_missing_ids_a = list(huri[huri['HGNC_A'].isnull()]['ENSG A'].unique())
huri_missing_ids_b = list(huri[huri['HGNC_B'].isnull()]['ENSG B'].unique())
huri_missing_ids = set(huri_missing_ids_a + huri_missing_ids_b)

ensg_ids_genecards, missed_ensg_genecards = id_conversion.genecards_search(huri_missing_ids, 'ensembl')
ncbi_search_hgnc_ids = id_conversion.ncbi_search(missed_ensg_genecards)
ensg_ids_genecards.update(ncbi_search_hgnc_ids)

for key, value in ensg_ids_genecards.items():
    huri.loc[huri['ENSG A'] == value[0], 'HGNC_A'] = key
    huri.loc[huri['ENSG B'] == value[0], 'HGNC_B'] = key

huri = huri[~((huri['HGNC_A'].isnull()) | (huri['HGNC_B'].isnull()))]
huri.to_csv('././data/interim/HuriHGNC_IDs.csv', header=-1, index=False)
