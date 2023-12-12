import os
import os.path as osp
from collections import Counter

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import transformers
import string
import nltk
from tqdm import tqdm
from nltk.corpus import stopwords
import logging
from torch_geometric.utils import k_hop_subgraph, subgraph, add_self_loops, to_undirected
from transformers import BertTokenizer

UNK = '[UNK]'
nltk.download('stopwords')
nltk.download('punkt')
STOP_WORDS = stopwords.words('english')
DROPPED = STOP_WORDS + list(string.punctuation)
CATEGORY_IDS = {'1-to-1': 0, '1-to-many': 1, 'many-to-1': 2, 'many-to-many': 3}

DEGREE_TRAIN = 'degree_train'
RANDOM = 'random'
TFIDF_RELEVANCE = 'tfidf_relevance'
SEMANTIC_RELEVANCE = 'semantic_relevance'
SEMANTIC_RELEVANCE_REDUNDANCY = 'semantic_relevance_redundancy'


def file_to_ids(file_path):
    """Read one line per file and assign it an ID.

    Args:
        file_path: str, path of file to read

    Returns: dict, mapping str to ID (int)
    """
    str2id = dict()
    with open(file_path) as file:
        for i, line in enumerate(file):
            str2id[line.strip()] = i

    return str2id


def get_negative_sampling_indices(batch_size, num_negatives, repeats=1):
    """"Obtain indices for negative sampling within a batch of entity pairs.
    Indices are sampled from a reshaped array of indices. For example,
    if there are 4 pairs (batch_size=4), the array of indices is
        [[0, 1],
         [2, 3],
         [4, 5],
         [6, 7]]
    From this array, we corrupt either the first or second element of each row.
    This yields one negative sample.
    For example, if the positions with a dash are selected,
        [[0, -],
         [-, 3],
         [4, -],
         [-, 7]]
    they are then replaced with a random index from a row other than the row
    to which they belong:
        [[0, 3],
         [5, 3],
         [4, 6],
         [1, 7]]
    The returned array has shape (batch_size, num_negatives, 2).
    """
    num_ents = batch_size * 2
    idx = torch.arange(num_ents).reshape(batch_size, 2)

    # For each row, sample entities, assigning 0 probability to entities
    # of the same row
    zeros = torch.zeros(batch_size, 2)
    head_weights = torch.ones(batch_size, num_ents, dtype=torch.float)
    head_weights.scatter_(1, idx, zeros)
    random_idx = head_weights.multinomial(num_negatives * repeats,
                                          replacement=True)
    random_idx = random_idx.t().flatten()

    # Select randomly the first or the second column
    row_selector = torch.arange(batch_size * num_negatives * repeats)
    col_selector = torch.randint(0, 2, [batch_size * num_negatives * repeats])

    # Fill the array of negative samples with the sampled random entities
    # at the right positions
    neg_idx = idx.repeat((num_negatives * repeats, 1))
    neg_idx[row_selector, col_selector] = random_idx
    neg_idx = neg_idx.reshape(-1, batch_size * repeats, 2)
    neg_idx.transpose_(0, 1)

    return neg_idx


class GraphDataset(Dataset):
    """A Dataset storing the triples of a Knowledge Graph.

    Args:
        triples_file: str, path to the file containing triples. This is a
            text file where each line contains a triple of the form
            'subject predicate object'
        write_maps_file: bool, if set to True, dictionaries mapping
            entities and relations to IDs are saved to disk (for reuse with
            other datasets).
    """

    def __init__(self, triples_file, neg_samples=None, write_maps_file=False,
                 num_devices=1):
        directory = osp.dirname(triples_file)
        maps_path = osp.join(directory, 'maps.pt')

        # Create or load maps from entity and relation strings to unique IDs
        if not write_maps_file:
            if not osp.exists(maps_path):
                raise ValueError('Maps file not found.')

            maps = torch.load(maps_path)
            ent_ids, rel_ids = maps['ent_ids'], maps['rel_ids']
        else:
            ents_file = osp.join(directory, 'entities.txt')
            rels_file = osp.join(directory, 'relations.txt')
            ent_ids = file_to_ids(ents_file)
            rel_ids = file_to_ids(rels_file)

        entities = set()
        relations = set()

        # Read triples and store as ints in tensor
        file = open(triples_file)
        triples = []
        for i, line in enumerate(file):
            values = line.split()
            # FB13 and WN11 have duplicate triples for classification,
            # here we keep the correct triple
            if len(values) > 3 and values[3] == '-1':
                continue
            head, rel, tail = line.split()[:3]
            entities.update([head, tail])
            relations.add(rel)
            triples.append([ent_ids[head], ent_ids[tail], rel_ids[rel]])

        self.triples = torch.tensor(triples, dtype=torch.long)

        self.rel_categories = torch.zeros(len(rel_ids), dtype=torch.long)
        rel_categories_file = osp.join(directory, 'relations-cat.txt')
        self.has_rel_categories = False
        if osp.exists(rel_categories_file):
            with open(rel_categories_file) as f:
                for line in f:
                    rel, cat = line.strip().split()
                    self.rel_categories[rel_ids[rel]] = CATEGORY_IDS[cat]
            self.has_rel_categories = True

        # Save maps for reuse
        torch.save({'ent_ids': ent_ids, 'rel_ids': rel_ids}, maps_path)

        self.num_ents = len(entities)
        self.num_rels = len(relations)
        self.entities = torch.tensor([ent_ids[ent] for ent in entities])
        self.num_triples = self.triples.shape[0]
        self.directory = directory
        self.maps_path = maps_path
        self.neg_samples = neg_samples
        self.num_devices = num_devices

    def __getitem__(self, index):
        return self.triples[index]

    def __len__(self):
        return self.num_triples

    def collate_fn(self, data_list):
        """Given a batch of triples, return it together with a batch of
        corrupted triples where either the subject or object are replaced
        by a random entity. Use as a collate_fn for a DataLoader.
        """
        batch_size = len(data_list)
        pos_pairs, rels = torch.stack(data_list).split(2, dim=1)
        neg_idx = get_negative_sampling_indices(batch_size, self.neg_samples)
        return pos_pairs, rels, neg_idx


class TextGraphDataset(GraphDataset):
    """A dataset storing a graph, and textual descriptions of its entities.

    Args:
        triples_file: str, path to the file containing triples. This is a
            text file where each line contains a triple of the form
            'subject predicate object'
        max_len: int, maximum number of tokens to read per description.
        neg_samples: int, number of negative samples to get per triple
        tokenizer: transformers.PreTrainedTokenizer or GloVeTokenizer, used
            to tokenize the text.
        drop_stopwords: bool, if set to True, punctuation and stopwords are
            dropped from entity descriptions.
        write_maps_file: bool, if set to True, dictionaries mapping
            entities and relations to IDs are saved to disk (for reuse with
            other datasets).
        drop_stopwords: bool
    """

    def __init__(self, triples_file, neg_samples, max_len, tokenizer,
                 drop_stopwords, write_maps_file=False, use_cached_text=False,
                 num_devices=1, tokenizer_name='bert-base-uncased'):
        super().__init__(triples_file, neg_samples, write_maps_file,
                         num_devices)
        self.logger = logging.getLogger()

        maps = torch.load(self.maps_path)
        ent_ids = maps['ent_ids']

        if max_len is None:
            max_len = tokenizer.max_len

        self.max_len = max_len

        cached_text_path = osp.join(self.directory, f'text_data_{tokenizer_name}.pt')
        self.logger.info(f'Cached text path: {cached_text_path}')
        need_to_load_text = True
        if use_cached_text:
            if osp.exists(cached_text_path):
                self.text_data = torch.load(cached_text_path)
                self.logger.info(f'Loaded cached text data for'
                                 f' {self.text_data.shape[0]} entities,'
                                 f' and maximum length {self.text_data.shape[1]}.')
                need_to_load_text = False
            else:
                self.logger.info(f'Cached text data not found.')

        if need_to_load_text:
            self.text_data = torch.zeros((len(ent_ids), max_len + 1),
                                         dtype=torch.long)
            read_entities = set()
            progress = tqdm(desc='Reading entity descriptions',
                            total=len(ent_ids), mininterval=5)
            for text_file in ('entity2textlong.txt', 'entity2text.txt'):
                file_path = osp.join(self.directory, text_file)
                if not osp.exists(file_path):
                    continue

                with open(file_path) as f:
                    for line in f:
                        values = line.strip().split('\t')
                        entity = values[0]
                        text = ' '.join(values[1:])
                        if entity not in ent_ids:
                            continue
                        if entity in read_entities:
                            continue

                        read_entities.add(entity)
                        ent_id = ent_ids[entity]

                        if drop_stopwords:
                            tokens = nltk.word_tokenize(text)
                            text = ' '.join([t for t in tokens if
                                             t.lower() not in DROPPED])

                        text_tokens = tokenizer.encode(text,
                                                       max_length=max_len,
                                                       return_tensors='pt')

                        text_len = text_tokens.shape[1]

                        # Starting slice of row contains token IDs
                        self.text_data[ent_id, :text_len] = text_tokens
                        # Last cell contains sequence length
                        self.text_data[ent_id, -1] = text_len

                        progress.update()

            progress.close()

            if len(read_entities) != len(ent_ids):
                raise ValueError(f'Read {len(read_entities):,} descriptions,'
                                 f' but {len(ent_ids):,} were expected.')

            if self.text_data[:, -1].min().item() < 1:
                # in the original code this raised an error
                # changed to print (results still look ok) to reproduce the experiments
                print(f'Some entries in text_data contain'
                      f' length-0 descriptions.')

            torch.save(self.text_data, cached_text_path)

        placeholder = torch.zeros((1, max_len + 1), dtype=torch.long)
        if isinstance(tokenizer, GloVeTokenizer):
            self.logger.info(f'Adding placeholder value for UNK token: {tokenizer.word2idx[UNK]}')
            placeholder[0][0] = tokenizer.word2idx[UNK]
        elif isinstance(tokenizer, BertTokenizer):
            self.logger.info(f'Adding placeholder value for UNK token: {tokenizer.unk_token_id}')
            placeholder[0][0] = tokenizer.unk_token_id

        placeholder[0][max_len] = 1  # length 1
        self.text_data = torch.cat((self.text_data, placeholder), dim=0)


    def get_entity_description(self, ent_ids):
        """Get entity descriptions for a tensor of entity IDs."""
        text_data = self.text_data[ent_ids]
        text_end_idx = text_data.shape[-1] - 1

        # Separate tokens from lengths
        text_tok, text_len = text_data.split(text_end_idx, dim=-1)
        max_batch_len = text_len.max()
        # Truncate batch
        text_tok = text_tok[..., :max_batch_len]
        text_mask = (text_tok > 0).float()

        return text_tok, text_mask, text_len

    def collate_fn(self, data_list):
        """Given a batch of triples, return it in the form of
        entity descriptions, and the relation types between them.
        Use as a collate_fn for a DataLoader.
        """
        batch_size = len(data_list) // self.num_devices
        if batch_size <= 1:
            raise ValueError('collate_text can only work with batch sizes'
                             ' larger than 1.')

        pos_pairs, rels = torch.stack(data_list).split(2, dim=1)
        text_tok, text_mask, text_len = self.get_entity_description(pos_pairs)

        neg_idx = get_negative_sampling_indices(batch_size, self.neg_samples,
                                                repeats=self.num_devices)

        return text_tok, text_mask, rels, neg_idx


class NeighborhoodTextGraphDataset(TextGraphDataset):
    def __init__(self, triples_file, neg_samples, max_len, tokenizer,
                 drop_stopwords, num_neighbors=5, selection=RANDOM, write_maps_file=False,
                 use_cached_text=False, use_cached_neighbors=False,
                 num_devices=1, device=None, tokenizer_name='bert-base-uncased', dataset_name='wikidata5m'):

        super().__init__(triples_file, neg_samples, max_len, tokenizer,
                         drop_stopwords, write_maps_file, use_cached_text,
                         num_devices, tokenizer_name)

        self.device = device
        self.num_neighbors = num_neighbors
        self.selection = selection

        maps = torch.load(self.maps_path)
        blp_ent_ids = maps['ent_ids']  # mapping: entity URI -> ID
        blp_rel_ids = maps['rel_ids']

        # as this class is just used for the training set, this list only contains train entity
        train_entities = self.entities

        # just for debugging
        if not use_cached_neighbors:

            self.load_page_link_graph_inductive(osp.join(self.directory, 'page_link_graph_typed.txt'),
                                                osp.join(self.directory, 'inductive_page_link_graph_typed.pt'),
                                                set(train_entities))

            self.page_link_graph_edge_index = torch.stack(
                [self.page_link_graph[:, 0], self.page_link_graph[:, 2]]).to(self.device)
            self.page_link_graph_edge_type = self.page_link_graph[:, 1].to(self.device)

            pre_computed_direct_neighborhood_file = osp.join(self.directory,
                                                             'pre_computed_direct_neighborhood.pt')

            if not osp.exists(pre_computed_direct_neighborhood_file):
                self.logger.info('Pre-computing neighborhood.')
                neighbor_dict = {}
                for id in tqdm(blp_ent_ids.values()):
                    _, edge_index_sample, _, _ = k_hop_subgraph(id, 1,
                                                                self.page_link_graph_edge_index,
                                                                relabel_nodes=False,
                                                                flow="target_to_source",
                                                                directed=True)
                    neighbor_dict[id] = torch.unique(edge_index_sample).cpu()
                torch.save(neighbor_dict, pre_computed_direct_neighborhood_file)
            else:
                self.logger.info('Loading pre-computed neighborhood.')
                neighbor_dict = torch.load(pre_computed_direct_neighborhood_file)

            max_relevant_neighbors = 20
            neighbor_tensor = torch.full((max(blp_ent_ids.values()) + 1, max_relevant_neighbors), -1)

            degree_dict = Counter(self.triples[:, 0].tolist() + self.triples[:, 1].tolist())

            for ent_id, neigh_ids in tqdm(neighbor_dict.items()):
                if self.selection == DEGREE_TRAIN:
                    neighbors_list_ordered = torch.tensor(sorted(neigh_ids.tolist(),
                                                    key=lambda e: - degree_dict[e]))
                else:
                    neighbors_list_ordered = neighbor_dict[ent_id]

                neighbor_tensor[ent_id, :min(neighbors_list_ordered.shape[0], max_relevant_neighbors)] = neighbors_list_ordered[
                                                                                            :max_relevant_neighbors]

            torch.save(neighbor_tensor, osp.join(self.directory, 'neighbors_tmp.pt'))
        else:
            self.logger.info('Attention: Loading pre-computed neighborhood for debugging.')
            neighbor_tensor = torch.load(osp.join(self.directory, 'neighbors_tmp.pt'))


        #self.neighbors = neighbor_tensor[:, :self.num_neighbors]
        self.neighbors = neighbor_tensor

    def get_neighbors(self, ent_ids, dynamic_selection='random'):
        """Get neighbors according to some dynamic selection strategy."""

        if dynamic_selection == 'random':
            neighbors = self.neighbors[ent_ids][ :, torch.randperm(10)[:self.num_neighbors]]
        elif dynamic_selection == 'no':
            neighbors = self.neighbors[:, :self.num_neighbors][ent_ids]
        else:
            raise ValueError('Unknown dynamic selection strategy:', dynamic_selection)
        return neighbors

    def collate_fn(self, data_list):
        """Given a batch of triples, return it in the form of
        entity descriptions, and the relation types between them.
        Use as a collate_fn for a DataLoader.
        """
        batch_size = len(data_list) // self.num_devices
        if batch_size <= 1:
            raise ValueError('collate_text can only work with batch sizes'
                             ' larger than 1.')

        pos_pairs, rels = torch.stack(data_list).split(2, dim=1)
        text_tok, text_mask, text_len = self.get_entity_description(pos_pairs)

        neighbors_head = self.get_neighbors(pos_pairs[:, 0])
        neighbors_tail = self.get_neighbors(pos_pairs[:, 1])

        text_tok_neighborhood_head, text_mask_neighborhood_head, _ = self.get_entity_description(
            neighbors_head)
        text_tok_neighborhood_tail, text_mask_neighborhood_tail, _ = self.get_entity_description(
            neighbors_tail)

        neg_idx = get_negative_sampling_indices(batch_size, self.neg_samples,
                                                repeats=self.num_devices)

        return text_tok, text_mask, rels, neg_idx, text_tok_neighborhood_head, text_mask_neighborhood_head, text_tok_neighborhood_tail, text_mask_neighborhood_tail

    def load_page_link_graph_inductive(self, raw_file_path, processed_file_path, train_entities):
        if os.path.isfile(processed_file_path):
            print('Loading inductive page link graph.')
            self.page_link_graph = torch.load(processed_file_path)
        else:
            maps = torch.load(self.maps_path)
            ent_uri_2_id = maps['ent_ids']
            rel_uri_2_id = maps['rel_ids']

            page_links_graph_raw = []
            skipped = 0
            print(raw_file_path)
            with open(raw_file_path) as page_linkes_raw_in:
                for line in page_linkes_raw_in:
                    head, relation, tail = line.strip().split('\t')
                    if head not in ent_uri_2_id or tail not in ent_uri_2_id:
                        skipped += 1
                        continue

                    page_links_graph_raw.append(
                        [ent_uri_2_id[head], rel_uri_2_id[relation], ent_uri_2_id[tail]])
            print('Skipped', skipped, 'triples because of missing entities.')

            page_links_graph_raw = torch.tensor(page_links_graph_raw, dtype=torch.long)

            mask_allowed_triples = []
            for i in tqdm(range(page_links_graph_raw.size(0))):
                head, relation, tail = page_links_graph_raw[i][0].item(), page_links_graph_raw[i][
                    1].item(), \
                                       page_links_graph_raw[i][2].item()
                # if an entity has a page link to a test entity, this link is removed if the entity itself is in
                # the training set. If the entity is also a test entity, there is no issue.
                if tail in train_entities:
                    mask_allowed_triples.append(True)
                else:
                    if head in train_entities:
                        mask_allowed_triples.append(False)
                    else:
                        mask_allowed_triples.append(True)

            print('Ratio of triples maintained:',
                  sum([1 for x in mask_allowed_triples if x]) / page_links_graph_raw.size(0))

            mask_allowed_triples = torch.tensor(mask_allowed_triples)
            page_links_graph_inductive = page_links_graph_raw[mask_allowed_triples]
            print('Page link graph filtered for the inductive setting.')
            torch.save(page_links_graph_inductive, processed_file_path)
            self.page_link_graph = page_links_graph_inductive

        self.page_link_graph = self.page_link_graph.type(torch.int64)


class GloVeTokenizer:
    def __init__(self, vocab_dict_file, uncased=True):
        self.word2idx = torch.load(vocab_dict_file)
        self.uncased = uncased

    def encode(self, text, max_length, return_tensors):
        if self.uncased:
            text = text.lower()
        tokens = nltk.word_tokenize(text)
        encoded = [self.word2idx.get(t, self.word2idx[UNK]) for t in tokens]
        encoded = [encoded[:max_length]]

        if return_tensors:
            encoded = torch.tensor(encoded)

        return encoded

    def batch_encode_plus(self, batch, max_length, **kwargs):
        batch_tokens = []
        for text in batch:
            tokens = self.encode(text, max_length, return_tensors=False)[0]
            if len(tokens) < max_length:
                tokens += [0] * (max_length - len(tokens))
            batch_tokens.append(tokens)

        batch_tokens = torch.tensor(batch_tokens, dtype=torch.long)
        batch_masks = (batch_tokens > 0).float()

        tokenized_data = {'input_ids': batch_tokens,
                          'attention_mask': batch_masks}

        return tokenized_data


def test_text_graph_dataset():
    from torch.utils.data import DataLoader

    tok = transformers.AlbertTokenizer.from_pretrained('albert-base-v2')
    gtr = TextGraphDataset('data/wikifb15k237/train-triples.txt', max_len=32,
                           neg_samples=32, tokenizer=tok, drop_stopwords=False)
    loader = DataLoader(gtr, batch_size=8, collate_fn=gtr.collate_fn)
    data = next(iter(loader))

    print('Done')


if __name__ == '__main__':
    test_text_graph_dataset()
