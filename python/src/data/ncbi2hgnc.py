import pandas as pd
import id_conversion
from tqdm import tqdm

hgnc_reference_table = pd.read_csv("././data/interim/HGNC symbols.txt", sep=',', header=0)
hgnc_reference_table.drop_duplicates(inplace=True)

ncbi_symbols = hgnc_reference_table[~hgnc_reference_table['NCBI Gene ID(supplied by NCBI)'].isnull()][['Approved symbol', 'NCBI Gene ID(supplied by NCBI)']]

reactome = pd.read_csv("././data/raw/NCBI2ReactomeReactions.txt", sep='\t', header=None, names = ['NCBI ID', 'Reactome ID', 'URL', 'Event', 'Evidence Code', 'Species'])
reactome = reactome[reactome['Species'] == 'Homo sapiens']

ncbi_reactome_id = list(reactome['NCBI ID'].unique())
for id in tqdm(ncbi_reactome_id):
    hgnc_id = hgnc_reference_table.loc[hgnc_reference_table['NCBI Gene ID(supplied by NCBI)'] == id, 'Approved symbol'].values
    if len(hgnc_id) > 0:
        reactome.loc[reactome['NCBI ID']==id, 'HGNC ID'] = hgnc_id[0]

missed_ncbi_ids = list(map(str, reactome[reactome['HGNC ID'].isnull()]['NCBI ID'].unique()))

ncbi_ids_genecards, missed_ncbi_genecards = id_conversion.genecards_search(missed_ncbi_ids, 'ncbi')
for key, value in ncbi_ids_genecards.items():
    reactome.loc[reactome['NCBI ID']==value[0], 'HGNC ID'] = key

reactome = reactome[~reactome['HGNC ID'].isnull()]
grouped = reactome.groupby(['Reactome ID'])
reactome = grouped.filter(lambda x: 50 <= len(x) <= 300)
reactome.to_csv('././data/interim/ReactomeReactions.csv', header=-1, index=False)
