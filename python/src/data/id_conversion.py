import requests
from tqdm.notebook import tqdm


def genecards_search(proteins, idtype='uniprot'):
    # Uses HGNC's REST web-service in order to convert uniprot, ensembl, ncbi or old HGNC IDs to current HGNC IDs.
    #
    # INPUT:
    #   -set of ids to convert
    #   -id format that the proteins are in
    #
    # RETURNS: one dict with the HGNC ids as keys and the original ones as values and a list with the ids that missed

    idtypes = {'uniprot': 'uniprot_ids', 'ensembl': 'ensembl_gene_id',
               'ncbi': 'entrez_id', 'alias symbol': 'alias_symbol'}
    id_map = {}
    no_match = []
    for id in tqdm(proteins):
        url = "http://rest.genenames.org/search/"+idtypes[idtype]+"/"+id
        response = requests.get(url, headers={'Accept': 'application/json'})
        json_info = response.json()
        if len(json_info['response']['docs']) == 1:
            for i in json_info['response']['docs']:
                hgnc = i['symbol']
                print(id, hgnc)
                if hgnc not in id_map.keys():
                    id_map[hgnc] = [id]
                else:
                    id_map[hgnc].append([id])
        else:
            print(id)
            no_match.append(id)
    return id_map, no_match


def ncbi_search(protein_ids):
    # Searches NCBI web-service to convert Ensembl and NCBI IDs to their HGNC GeneName
    #
    # INPUT:
    #   -set of ids to convert
    #
    # RETURNS: one dict with the HGNC ids as keys and the original ones as values
    id_map = {}
    for id in tqdm(protein_ids):
        response = requests.get(
            "https://www.ncbi.nlm.nih.gov/gene/?term=" + id + "&report=full_report&format=text")
        if len(response.text) <= 172:
            print(id)
        else:
            data = response.text.splitlines()
            for line in data:
                if line.startswith('Official Symbol:') and line.endswith('(provided by HGNC)'):
                    hgnc_code = line.split(' ')[2]
                    id_map[hgnc_code] = [id]
                    print(id, hgnc_code)
                    continue
            if [id] not in id_map.values():
                print(id)
    return id_map


def confirm_hgnc(protein_list):
    # Uses HGNC's REST web-service in order to confirm is the HGNC IDs obtained in other sources are the current ones,
    # or if they're aliases, older symbols or names of the gene while retriving their updated ID.
    #
    # INPUT:
    #   -set of ids to convert
    #
    # RETURNS: one dict with the HGNC ids as keys with the original ones as values, and a list with the unresolved ones.
    idtypes = {'symbol': 'symbol', 'previous symbol': 'prev_symbol', 'name': 'name', 'alias symbol': 'alias_symbol',
               'previous name': 'prev_name',  'alias names': 'alias_name'}
    resolved_ids = {}
    for idtype in idtypes.values():
        if len(protein_list) == 0:
            break
        print('Getting results for {}'.format(idtype))
        print('Testing {} proteins'.format(len(protein_list)))
        print('Number of resolved IDs: {}'.format(len(resolved_ids)))
        proteins_to_remove = []
        for protein in tqdm(protein_list):
            if '/' in protein:

                if protein not in resolved_ids.keys():
                    resolved_ids[protein] = [protein]
                else:
                    resolved_ids[protein].extend([protein])
                proteins_to_remove.append(protein)
            else:
                url = "http://rest.genenames.org/search/"+idtype+"/"+protein
                response = requests.get(url, headers={'Accept': 'application/json'})
                json_info = response.json()
                if len(json_info['response']['docs']) == 1 and idtype == 'symbol':
                    for i in json_info['response']['docs']:
                        hgnc = i['symbol']
                        print(protein, hgnc)
                        if hgnc not in resolved_ids.keys():
                            resolved_ids[hgnc] = [protein]
                        else:
                            resolved_ids[hgnc].extend([protein])
                        proteins_to_remove.append(protein)
                elif 0 < len(json_info['response']['docs']) < 4 and idtype != 'symbol':
                    protein_info = json_info['response']['docs'][0]
                    hgnc = protein_info['symbol']
                    print(protein, hgnc)
                    if hgnc not in resolved_ids.keys():
                        resolved_ids[hgnc] = [protein]
                    else:
                        resolved_ids[hgnc].extend([protein])
                    proteins_to_remove.append(protein)
        protein_list = [protein for protein in protein_list if protein not in proteins_to_remove]
    return resolved_ids, protein_list

