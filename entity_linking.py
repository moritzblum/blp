import spacy
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="Wikidata5M", type=str, help="Dataset to use, choose from: Wikidata5M, FB15k-237")
args = parser.parse_args()

dataset = args.dataset
print('Linking:', dataset)
dataset_path = f'./data/{dataset}/entity2textlong.txt'

# initialize language model
nlp = spacy.load("en_core_web_md")

# add pipeline (declared through entry_points in setup.py)
nlp.add_pipe("entityLinker", last=True)


if 'FB' in dataset_path:
    # Load Wikidata -> Freebase mapping
    entity2wikidata = json.load(open('./data/FB15k-237/entity2wikidata.json'))

    wikidata2freebase = {}
    for freebase_id in entity2wikidata.keys():
        wikidata_id = entity2wikidata[freebase_id]['wikidata_id']
        wikidata2freebase[wikidata_id] = freebase_id


num_entities = sum(1 for _ in open(dataset_path))

with open(dataset_path) as text_in:

    page_links = {}

    for index, line in enumerate(tqdm(text_in, total=num_entities)):

        uri, description = line.split('\t', 1)
        page_links[uri] = []

        doc = nlp(description)
        all_linked_entities = doc._.linkedEntities

        for entity in all_linked_entities:
            subj_uri = 'Q' + str(entity.get_id())

            if 'FB' in dataset_path:
                # for freebased entities, map linked wikidata ids to freebase ids
                if subj_uri in wikidata2freebase:
                    page_links[uri].append(wikidata2freebase[subj_uri])
            else:
                page_links[uri].append(subj_uri)


json.dump(page_links, open(f'./data/{dataset}/page_links.json', 'w'))
