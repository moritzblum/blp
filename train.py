import os
import os.path as osp
import random
import string
import time
from datetime import datetime

import networkx as nx
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from sacred.run import Run
from logging import Logger
from sacred import Experiment
from sacred.observers import MongoObserver
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import joblib

from data import CATEGORY_IDS, RANDOM, get_negative_sampling_indices, ATTENTION
from data import GraphDataset, TextGraphDataset, GloVeTokenizer, NeighborhoodTextGraphDataset
import models
import utils
import wandb

from models import NeighborhoodSelectionTransformer

OUT_PATH = 'output/'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ex = Experiment()
ex.logger = utils.get_logger()
# Set up database logs
uri = os.environ.get('DB_URI')
database = os.environ.get('DB_NAME')
if all([uri, database]):
    ex.observers.append(MongoObserver(uri, database))

"""
# -- comments ---
* env (GPU Cluster): lp



# --- train model ---
# baseline
python train.py -u link_prediction with dataset='FB15k-237' inductive=True dim=128 model='blp' rel_model='transe' loss_fn='margin' encoder_name='bert-base-cased' regularizer=0 max_len=32 num_negatives=64 lr=2e-5 use_scheduler=True batch_size=64 emb_batch_size=512 eval_batch_size=64 max_epochs=40 use_cached_text=True neighborhood_enrichment=False wandb_logging=True
# random neighborhood selection 
python train.py -u link_prediction with dataset='FB15k-237' inductive=True dim=128 model='blp' rel_model='transe' loss_fn='margin' encoder_name='bert-base-cased' regularizer=0 max_len=32 num_negatives=64 lr=2e-5 use_scheduler=True batch_size=64 emb_batch_size=512 eval_batch_size=64 max_epochs=40 use_cached_text=True neighborhood_enrichment=True use_cached_neighbors=True num_neighbors=3 neigh_ordering='degree_train' neigh_selection='random' wandb_logging=True
# neighborhood selection by deegree
python train.py -u link_prediction with dataset='FB15k-237' inductive=True dim=128 model='blp' rel_model='transe' loss_fn='margin' encoder_name='bert-base-cased' regularizer=0 max_len=32 num_negatives=64 lr=2e-5 use_scheduler=True batch_size=64 emb_batch_size=512 eval_batch_size=64 max_epochs=40 use_cached_text=True neighborhood_enrichment=True use_cached_neighbors=True num_neighbors=3 neigh_ordering='degree_train' neigh_selection='no' wandb_logging=True
# attention neighborhood selection
python train.py -u link_prediction with dataset='FB15k-237' inductive=True dim=128 model='blp' rel_model='transe' loss_fn='margin' encoder_name='bert-base-cased' regularizer=0 max_len=32 num_negatives=64 lr=2e-5 use_scheduler=True batch_size=64 emb_batch_size=512 eval_batch_size=64 max_epochs=40 use_cached_text=True neighborhood_enrichment=True use_cached_neighbors=True num_neighbors=3 neigh_ordering='degree_train' neigh_selection='attention' wandb_logging=True


"""


@ex.config
def config():
    dataset = 'umls'
    inductive = True
    dim = 128
    model = 'blp'
    rel_model = 'transe'
    loss_fn = 'margin'
    encoder_name = 'bert-base-cased'
    regularizer = 0
    max_len = 32
    num_negatives = 64
    lr = 2e-5
    use_scheduler = True
    batch_size = 64
    emb_batch_size = 512
    eval_batch_size = 64
    max_epochs = 40
    checkpoint = None
    checkpoint_neigh = None
    use_cached_text = False
    use_cached_neighbors = False
    neighborhood_enrichment = False
    weighted_pooling = False
    # neighborhood selection strategy
    num_neighbors = 5
    neigh_ordering = RANDOM
    neigh_selection = ATTENTION
    # logging
    wandb_logging = False
    edge_features = False


@ex.capture
@torch.no_grad()
def eval_link_prediction_neigh_attention_avg_loss(model, triples_loader, text_dataset, _log, num_negative,
                                                  prefix='test', neighborhood_enrichment=True,
                                                  neigh_selection='attention'):
    eval_loss_all = 0
    _log.info(f'Evaluating Loss on {prefix} set.')

    for i, triples in enumerate(triples_loader):
        heads, tails, rels = torch.chunk(triples, chunks=3, dim=1)

        heads = heads.flatten()
        tails = tails.flatten()
        rels = rels.flatten()

        text_tok_head, text_mask_head, _ = text_dataset.get_entity_description(heads)
        text_tok_tail, text_mask_tail, _ = text_dataset.get_entity_description(tails)

        if not neighborhood_enrichment:
            batch_emb_head = model(text_tok=text_tok_head.to(device),
                                   text_mask=text_mask_head.to(device),
                                   text_tok_neighborhood_all=None)

            batch_emb_tail = model(text_tok=text_tok_tail.to(device),
                                   text_mask=text_mask_tail.to(device),
                                   text_tok_neighborhood_all=None)

        else:
            neighbors_head = text_dataset.get_neighbors(heads, rels=None,
                                                        neigh_selection=neigh_selection)
            neighbors_tail = text_dataset.get_neighbors(tails, rels=None,
                                                        neigh_selection=neigh_selection)

            text_tok_neighbors_head, _, _ = text_dataset.get_entity_description(
                neighbors_head)
            text_tok_neighbors_tail, _, _ = text_dataset.get_entity_description(
                neighbors_tail)

            batch_emb_head = model(
                text_tok=text_tok_head.reshape((-1, text_tok_head.size(-1))).to(device),
                text_mask=text_mask_head.reshape((-1, text_mask_head.size(-1))).to(device),
                text_tok_neighborhood_all=text_tok_neighbors_head.to(device))

            batch_emb_tail = model(
                text_tok=text_tok_tail.reshape((-1, text_tok_tail.size(-1))).to(device),
                text_mask=text_mask_tail.reshape((-1, text_mask_tail.size(-1))).to(device),
                text_tok_neighborhood_all=text_tok_neighbors_tail.to(device))

        neg_idx = get_negative_sampling_indices(batch_emb_head.size(0), num_negative)

        eval_loss = model.module.compute_loss(torch.stack((batch_emb_head, batch_emb_tail), dim=1),
                                              rels.unsqueeze(dim=1).to(device), neg_idx).mean()
        eval_loss_all += eval_loss.item()

    avg_eval_loss = eval_loss_all / len(triples_loader)

    _log.info(f'Avg. {prefix} loss: {avg_eval_loss:.4f}')

    return avg_eval_loss


@ex.capture
@torch.no_grad()
def eval_link_prediction_neigh_attention(model, triples_loader, text_dataset, entities,
                                         _run: Run, _log: Logger,
                                         prefix='', max_num_batches=None,
                                         filtering_graph=None, new_entities=None,
                                         neighborhood_selector=None,
                                         emb_dim=128, neigh_selection='attention'):
    if neighborhood_selector is None:
        _log.info(f'Starting evaluation > without < neighborhood attention on {prefix} set.')
    else:
        _log.info(f'Starting evaluation > with < neighborhood attention on {prefix} set.')

    compute_filtered = filtering_graph is not None
    _log.info(f'Computing the filtered setting: {compute_filtered}.')

    # evaluation code
    mrr_by_position = torch.zeros(3, dtype=torch.float).to(device)
    mrr_pos_counts = torch.zeros_like(mrr_by_position)

    rel_categories = triples_loader.dataset.rel_categories.to(device)
    mrr_by_category = torch.zeros([2, 4], dtype=torch.float).to(device)
    mrr_cat_count = torch.zeros([1, 4], dtype=torch.float).to(device)

    hit_positions = [1, 3, 10]
    k_values = torch.tensor([hit_positions], device=device)
    hits_at_k = {pos: 0.0 for pos in hit_positions}
    mrr = 0.0
    mrr_filt = 0.0
    hits_at_k_filt = {pos: 0.0 for pos in hit_positions}

    num_relations = model.module.num_relations

    if device != torch.device('cpu'):
        model = model.module

    num_entities = entities.shape[0]

    if compute_filtered:
        max_ent_id = max(filtering_graph.nodes)
    else:
        max_ent_id = entities.max()

    _log.info('Computing embeddings.')
    ent2idx = torch.full([max_ent_id + 1], fill_value=-1, dtype=torch.long)
    ent_emb = torch.full((num_entities, emb_dim), fill_value=-1, dtype=torch.float)

    for ent_idx, ent_id in enumerate(entities.tolist()):
        ent2idx[ent_id] = ent_idx

        if neighborhood_selector is None:
            text_tok, text_mask, _ = text_dataset.get_entity_description(torch.tensor([ent_id]))
            text_tok_neighbors = None
        else:
            text_tok, text_mask, _ = text_dataset.get_entity_description(torch.tensor([ent_id]))

            neighbors = neighborhood_selector(torch.tensor([ent_id]), rels=None,
                                              neigh_selection=neigh_selection)
            text_tok_neighbors, text_mask_neighbors, _ = text_dataset.get_entity_description(neighbors)
            text_tok_neighbors = text_tok_neighbors.to(device)

        text_len = text_tok.size(1)
        batch_emb_id = model(text_tok=text_tok.reshape((-1, text_len)).to(device),
                             text_mask=text_mask.reshape((-1, text_len)).to(device),
                             text_tok_neighborhood_all=text_tok_neighbors)

        ent_emb[ent_idx] = batch_emb_id

    num_predictions = 0
    _log.info('Computing metrics on set of triples')
    total = len(triples_loader) if max_num_batches is None else max_num_batches

    triple_ranks_all = []
    triple_ranks_filtered_all = []
    for i, triples in enumerate(triples_loader):

        if max_num_batches is not None and i == max_num_batches:
            break

        heads, tails, rels = torch.chunk(triples, chunks=3, dim=1)
        t = torch.cat((heads, rels, tails), dim=1)

        # Map entity IDs to positions in ent_emb
        heads = ent2idx[heads.flatten()]
        tails = ent2idx[tails.flatten()]

        assert heads.min() >= 0
        assert tails.min() >= 0

        # Embed triples
        # shape: batch size x emb_dim
        head_embs = ent_emb[heads]
        tail_embs = ent_emb[tails]

        ent_emb_new = ent_emb.repeat((tail_embs.size(0), 1, 1)).to(device)

        tail_embs = tail_embs.unsqueeze(1).repeat((1, ent_emb_new.size(1), 1)).to(device)
        head_embs = head_embs.unsqueeze(1).repeat((1, ent_emb_new.size(1), 1)).to(device)
        rel_embs = model.rel_emb(rels.flatten().repeat(ent_emb_new.size(1), 1).T.to(device))

        heads_predictions = model.score_fn(ent_emb_new, tail_embs, rel_embs, eval=True)
        tails_predictions = model.score_fn(head_embs, ent_emb_new, rel_embs, eval=True)

        pred_ents = torch.cat((heads_predictions, tails_predictions))

        true_ents = torch.cat((heads.flatten().unsqueeze(1),
                               tails.flatten().unsqueeze(1))).to(device)

        num_predictions += pred_ents.shape[0]
        reciprocals, hits, ranks = utils.get_metrics(pred_ents, true_ents, k_values)

        # collect triples and their ranks for later analysis
        ranks = torch.reshape(ranks, (2, -1))  # split head and tail ranks
        triple_ranks = torch.cat(
            (t, ranks[0].unsqueeze(1).cpu(), ranks[1].unsqueeze(1).cpu()), dim=1).type(torch.int32)

        triple_ranks_all.append(triple_ranks)

        mrr += reciprocals.sum().item()
        hits_sum = hits.sum(dim=0)

        for j, k in enumerate(hit_positions):
            hits_at_k[k] += hits_sum[j].item()

        if compute_filtered:
            filters = utils.get_triple_filters(triples, filtering_graph, num_entities, ent2idx)
            heads_filter, tails_filter = filters
            # Filter entities by assigning them the lowest score in the batch
            filter_mask = torch.cat((heads_filter, tails_filter)).to(device)
            pred_ents[filter_mask] = pred_ents.min() - 1.0

            reciprocals, hits, ranks = utils.get_metrics(pred_ents, true_ents, k_values)

            # collect triples and their ranks for later analysis
            ranks = torch.reshape(ranks, (2, -1))  # split head and tail ranks
            triple_ranks_filtered = torch.cat(
                (t, ranks[0].unsqueeze(1).cpu(), ranks[1].unsqueeze(1).cpu()), dim=1).type(torch.int32)
            triple_ranks_filtered_all.append(triple_ranks_filtered)

            mrr_filt += reciprocals.sum().item()
            hits_sum = hits.sum(dim=0)
            for j, k in enumerate(hit_positions):
                hits_at_k_filt[k] += hits_sum[j].item()

            reciprocals = reciprocals.squeeze()
            if new_entities is not None:
                by_position = utils.split_by_new_position(triples,
                                                          reciprocals,
                                                          new_entities)
                batch_mrr_by_position, batch_mrr_pos_counts = by_position
                mrr_by_position += batch_mrr_by_position
                mrr_pos_counts += batch_mrr_pos_counts

            if triples_loader.dataset.has_rel_categories:
                by_category = utils.split_by_category(triples,
                                                      reciprocals,
                                                      rel_categories)
                batch_mrr_by_cat, batch_mrr_cat_count = by_category
                mrr_by_category += batch_mrr_by_cat
                mrr_cat_count += batch_mrr_cat_count

        if (i + 1) % int(0.2 * total) == 0:
            _log.info(f'[{i + 1:,}/{total:,}]')

    _log.info(f'The total number of predictions is {num_predictions:,}')

    for hits_dict in (hits_at_k, hits_at_k_filt):
        for k in hits_dict:
            hits_dict[k] /= num_predictions
    mrr = mrr / num_predictions

    triple_ranks_all = torch.cat(triple_ranks_all, dim=0)
    mr = triple_ranks_all[:, 3:].float().mean().item()

    scores = {}
    scores['mrr'] = mrr
    scores['mr'] = mr

    log_str = f'{prefix} mrr: {mrr:.4f}  '
    log_str += f'{prefix} mr: {mr:.4f}  '

    for k, value in hits_at_k.items():
        scores[f'hits@{k}'] = value
        log_str += f'hits@{k}: {value:.4f}  '

    if compute_filtered:
        mrr_filt = mrr_filt / num_predictions
        triple_ranks_filtered_all = torch.cat(triple_ranks_filtered_all, dim=0)
        mr_filt = triple_ranks_filtered_all[:, 3:].float().mean().item()

        scores['mrr_filt'] = mrr_filt
        scores['mr_filt'] = mr_filt

        log_str += f'mrr_filt: {mrr_filt:.4f}  '
        log_str += f'mr_filt: {mr_filt:.4f}  '

        for k, value in hits_at_k_filt.items():
            scores[f'hits@{k}_filt'] = value
            log_str += f'hits@{k}_filt: {value:.4f}  '

    # logging all default metrics to stdout
    _log.info(log_str)

    if new_entities is not None and compute_filtered:
        mrr_pos_counts[mrr_pos_counts < 1.0] = 1.0
        mrr_by_position = mrr_by_position / mrr_pos_counts
        log_str = ''
        for i, t in enumerate((f'{prefix}_mrr_filt_both_new',
                               f'{prefix}_mrr_filt_head_new',
                               f'{prefix}_mrr_filt_tail_new')):
            value = mrr_by_position[i].item()
            log_str += f'{t}: {value:.4f}  '
        _log.info(log_str)

    if compute_filtered and triples_loader.dataset.has_rel_categories:
        mrr_cat_count[mrr_cat_count < 1.0] = 1.0
        mrr_by_category = mrr_by_category / mrr_cat_count

        for i, case in enumerate(['pred_head', 'pred_tail']):
            log_str = f'{case} '
            for cat, cat_id in CATEGORY_IDS.items():
                log_str += f'{cat}_mrr: {mrr_by_category[i, cat_id]:.4f}  '
            _log.info(log_str)

    return scores


@ex.command
def link_prediction(dataset, inductive, dim, model, rel_model, loss_fn,
                    encoder_name, regularizer, max_len, num_negatives, lr,
                    use_scheduler, batch_size, emb_batch_size, eval_batch_size,
                    max_epochs, checkpoint, checkpoint_neigh, use_cached_text, use_cached_neighbors,
                    neighborhood_enrichment, num_neighbors, edge_features,
                    weighted_pooling, neigh_ordering, neigh_selection, wandb_logging,
                    _run: Run, _log: Logger):
    _log.info(f'config: {_run.config}')

    if checkpoint is not None:
        _log.info(f'Evaluation. Checkpoint: {checkpoint}, {checkpoint_neigh}')

    if wandb_logging and checkpoint is None:
        wandb.login()
        wandb.init(
            # Set the project where this run will be logged
            project="blp",
            # Track hyperparameters and run metadata
            config=_run.config)

    drop_stopwords = model in {'bert-bow', 'bert-dkrl',
                               'glove-bow', 'glove-dkrl'}

    prefix = 'ind-' if inductive and model != 'transductive' else ''
    triples_file = f'data/{dataset}/{prefix}train.tsv'

    if device != torch.device('cpu'):
        num_devices = torch.cuda.device_count()
        if batch_size % num_devices != 0:
            raise ValueError(f'Batch size ({batch_size}) must be a multiple of'
                             f' the number of CUDA devices ({num_devices})')
        _log.info(f'CUDA devices used: {num_devices}')
    else:
        num_devices = 1
        _log.info('Training on CPU')

    if model == 'transductive':
        train_data = GraphDataset(triples_file, num_negatives,
                                  write_maps_file=True,
                                  num_devices=num_devices)
    else:
        if model.startswith('bert') or model == 'blp':
            tokenizer = BertTokenizer.from_pretrained(encoder_name)
            tokenizer_name = encoder_name
        else:
            tokenizer = GloVeTokenizer('data/glove/glove.6B.300d-maps.pt')
            tokenizer_name = 'glove'

        print('neighborhood_enrichment:', neighborhood_enrichment)
        if neighborhood_enrichment:
            train_data = NeighborhoodTextGraphDataset(triples_file,
                                                      num_negatives,
                                                      max_len,
                                                      tokenizer,
                                                      drop_stopwords,
                                                      num_neighbors=num_neighbors,
                                                      neigh_ordering=neigh_ordering,
                                                      neigh_selection=neigh_selection,
                                                      write_maps_file=True,
                                                      use_cached_text=use_cached_text,
                                                      use_cached_neighbors=use_cached_neighbors,
                                                      num_devices=num_devices,
                                                      device=device,
                                                      tokenizer_name=tokenizer_name)
        else:
            train_data = TextGraphDataset(triples_file,
                                          num_negatives,
                                          max_len,
                                          tokenizer,
                                          drop_stopwords,
                                          write_maps_file=True,
                                          use_cached_text=use_cached_text,
                                          num_devices=num_devices,
                                          tokenizer_name=tokenizer_name)

    train_loader = DataLoader(train_data, batch_size, shuffle=True,
                              collate_fn=train_data.collate_fn,
                              num_workers=0, drop_last=True)

    train_eval_loader = DataLoader(train_data, eval_batch_size)

    _log.info(f'Loading validation data: data/{dataset}/{prefix}dev.tsv')
    valid_data = GraphDataset(f'data/{dataset}/{prefix}dev.tsv')
    valid_loader = DataLoader(valid_data, eval_batch_size)

    _log.info(f'Loading validation data: data/{dataset}/{prefix}test.tsv')
    test_data = GraphDataset(f'data/{dataset}/{prefix}test.tsv')
    test_loader = DataLoader(test_data, eval_batch_size)

    # Build graph with all triples to compute filtered metrics
    if dataset != 'Wikidata5M':
        graph = nx.MultiDiGraph()
        all_triples = torch.cat((train_data.triples,
                                 valid_data.triples,
                                 test_data.triples))
        graph.add_weighted_edges_from(all_triples.tolist())

        train_ent = set(train_data.entities.tolist())
        train_val_ent = set(valid_data.entities.tolist()).union(train_ent)
        train_val_test_ent = set(test_data.entities.tolist()).union(train_val_ent)
        val_new_ents = train_val_ent.difference(train_ent)
        test_new_ents = train_val_test_ent.difference(train_val_ent)

    else:
        graph = None

        train_ent = set(train_data.entities.tolist())
        train_val_ent = set(valid_data.entities.tolist())
        train_val_test_ent = set(test_data.entities.tolist())
        val_new_ents = test_new_ents = None

    _run.log_scalar('num_train_entities', len(train_ent))

    train_ent = torch.tensor(list(train_ent))
    train_val_ent = torch.tensor(list(train_val_ent))
    train_val_test_ent = torch.tensor(list(train_val_test_ent))

    model = utils.get_model(model, dim, rel_model, loss_fn,
                            len(train_val_test_ent), train_data.num_rels,
                            encoder_name, regularizer, num_neighbors, edge_features,
                            weighted_pooling)

    if neighborhood_enrichment and neigh_selection == ATTENTION:
        model_neigh_sel = NeighborhoodSelectionTransformer(encoder_name=encoder_name)
    else:
        model_neigh_sel = None

    if checkpoint is not None:
        _log.info(f'Loading model from checkpoint: {checkpoint}, {checkpoint_neigh}')
        model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
        if model_neigh_sel is not None:
            model_neigh_sel.load_state_dict(torch.load(checkpoint_neigh, map_location='cpu'))
        _log.info('Models loaded')

    if device != torch.device('cpu'):
        model = torch.nn.DataParallel(model).to(device)
        if model_neigh_sel is not None:
            model_neigh_sel = torch.nn.DataParallel(model_neigh_sel).to(device)

    # provide the data loader with the model for the neighborhood attention
    if model_neigh_sel is not None:
        train_data.neighborhood_attention_model = model_neigh_sel

    if checkpoint is None:
        _log.info('Start model training')
        if model_neigh_sel is not None:
            optimizer = Adam(list(model.parameters()) + list(model_neigh_sel.parameters()), lr=lr)
        else:
            optimizer = Adam(model.parameters(), lr=lr)
        total_steps = len(train_loader) * max_epochs
        if use_scheduler:
            warmup = int(0.2 * total_steps)
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=warmup,
                                                        num_training_steps=total_steps)
        best_valid_mrr = 0.0

        checkpoint_file = osp.join(OUT_PATH,
                                   f'model-{encoder_name}-{datetime.now().strftime("%Y-%m-%d-%H-%M")}-{"".join([random.choice(string.ascii_lowercase) for _ in range(6)])}')
        _log.info(f'checkpoint_file: {checkpoint_file} + .pt/_neigh_sel.pt')

        epoch_start_time = time.time()
        for epoch in range(1, max_epochs + 1):
            train_loss = 0

            for step, data in enumerate(train_loader):
                batch_start_time = time.time()

                loss = model(*data).mean()

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                if use_scheduler:
                    scheduler.step()

                train_loss += loss.item()

                if step % int(0.05 * len(train_loader)) == 0:
                    _log.info(f'Epoch {epoch}/{max_epochs} '
                              f'[{step}/{len(train_loader)}]: {loss.item():.6f}')
                    _run.log_scalar('batch_loss', loss.item())

                # _log.info(f'Time per batch: {time.time() - batch_start_time}')

            _run.log_scalar('train_loss', train_loss / len(train_loader), epoch)
            _log.info(f'Time per epoch: {time.time() - epoch_start_time}')

            val_scores = eval_link_prediction_neigh_attention(model,
                                                              test_loader,
                                                              train_data,
                                                              train_val_test_ent,
                                                              prefix='test',
                                                              filtering_graph=graph,
                                                              new_entities=test_new_ents,
                                                              neighborhood_selector=train_data.get_neighbors if neighborhood_enrichment else None,
                                                              emb_dim=dim,
                                                              neigh_selection=neigh_selection)


            # Keep checkpoint of best performing model (based on raw MRR)
            if val_scores['mrr_filt'] > best_valid_mrr:
                best_valid_mrr = val_scores['mrr_filt']
                torch.save(model.module.state_dict(), checkpoint_file + '.pt')
                if model_neigh_sel is not None:
                    torch.save(model_neigh_sel.module.state_dict(), checkpoint_file + '_neigh_sel.pt')

                if wandb_logging:
                    log_data = val_scores

                    avg_loss = train_loss / len(train_loader)
                    avg_eval_loss_ = eval_link_prediction_neigh_attention_avg_loss(model, valid_loader, train_data,
                                                                                   _log=_log, num_negative=num_negatives,
                                                                                   prefix='valid', neighborhood_enrichment=neighborhood_enrichment, neigh_selection=neigh_selection)
                    avg_test_loss_ = eval_link_prediction_neigh_attention_avg_loss(model, test_loader, train_data,
                                                                                   _log=_log, num_negative=num_negatives,
                                                                                   prefix='test', neighborhood_enrichment=neighborhood_enrichment, neigh_selection=neigh_selection)

                    log_data["loss"] = avg_loss
                    log_data["eval_loss"] = avg_eval_loss_
                    log_data["test_loss"] = avg_test_loss_
                    wandb.log(log_data, step=epoch)

        # Evaluate with best performing checkpoint
        if max_epochs > 0:
            _log.info(f'--> Loading best model from checkpoint: {checkpoint_file}')
            model.module.load_state_dict(torch.load(checkpoint_file + '.pt'))
            if model_neigh_sel is not None:
                model_neigh_sel.module.load_state_dict(torch.load(checkpoint_file + '_neigh_sel.pt'))

    if dataset == 'Wikidata5M':
        graph = nx.MultiDiGraph()
        graph.add_weighted_edges_from(valid_data.triples.tolist())

    _log.info('Evaluating on validation set')
    eval_start_time = time.time()
    eval_link_prediction_neigh_attention(model,
                                         valid_loader,
                                         train_data,
                                         train_val_ent,
                                         prefix='valid',
                                         filtering_graph=graph,
                                         new_entities=val_new_ents,
                                         neighborhood_selector=train_data.get_neighbors if neighborhood_enrichment else None,
                                         emb_dim=dim, 
                                         neigh_selection=neigh_selection)
    _log.info(f'Evaluation time: {time.time() - eval_start_time}')

    if dataset == 'Wikidata5M':
        graph = nx.MultiDiGraph()
        graph.add_weighted_edges_from(test_data.triples.tolist())

    _log.info('Evaluating on test set')
    eval_start_time = time.time()
    eval_link_prediction_neigh_attention(model,
                                         test_loader,
                                         train_data,
                                         train_val_test_ent,
                                         prefix='test',
                                         filtering_graph=graph,
                                         new_entities=test_new_ents,
                                         neighborhood_selector=train_data.get_neighbors if neighborhood_enrichment else None,
                                         emb_dim=dim,
                                         neigh_selection=neigh_selection)
    _log.info(f'Evaluation time: {time.time() - eval_start_time}')


@ex.command
def node_classification(dataset, checkpoint, _run: Run, _log: Logger):
    ent_emb = torch.load(f'output/ent_emb-{checkpoint}.pt', map_location='cpu')
    if isinstance(ent_emb, tuple):
        ent_emb = ent_emb[0]

    ent_emb = ent_emb.squeeze().numpy()
    num_embs, emb_dim = ent_emb.shape
    _log.info(f'Loaded {num_embs} embeddings with dim={emb_dim}')

    emb_ids = torch.load(f'output/ents-{checkpoint}.pt', map_location='cpu')
    ent2idx = utils.make_ent2idx(emb_ids, max_ent_id=emb_ids.max()).numpy()
    maps = torch.load(f'data/{dataset}/maps.pt')
    ent_ids = maps['ent_ids']
    class2label = defaultdict(lambda: len(class2label))

    splits = ['train', 'dev', 'test']
    split_2data = dict()
    for split in splits:
        with open(f'data/{dataset}/{split}-ents-class.txt') as f:
            idx = []
            labels = []
            for line in f:
                entity, ent_class = line.strip().split()
                entity_id = ent_ids[entity]
                entity_idx = ent2idx[entity_id]
                idx.append(entity_idx)
                labels.append(class2label[ent_class])

            x = ent_emb[idx]
            y = np.array(labels)
            split_2data[split] = (x, y)

    x_train, y_train = split_2data['train']
    x_dev, y_dev = split_2data['dev']
    x_test, y_test = split_2data['test']

    best_dev_metric = 0.0
    best_c = 0
    for k in range(-4, 2):
        c = 10 ** -k
        model = LogisticRegression(C=c, multi_class='multinomial',
                                   max_iter=1000)
        model.fit(x_train, y_train)

        dev_preds = model.predict(x_dev)
        dev_acc = accuracy_score(y_dev, dev_preds)
        _log.info(f'{c:.3f} - {dev_acc:.3f}')

        if dev_acc > best_dev_metric:
            best_dev_metric = dev_acc
            best_c = c

    _log.info(f'Best regularization coefficient: {best_c:.4f}')
    model = LogisticRegression(C=best_c, multi_class='multinomial',
                               max_iter=1000)
    x_train_all = np.concatenate((x_train, x_dev))
    y_train_all = np.concatenate((y_train, y_dev))
    model.fit(x_train_all, y_train_all)

    for metric_fn in (accuracy_score, balanced_accuracy_score):
        train_preds = model.predict(x_train_all)
        train_metric = metric_fn(y_train_all, train_preds)

        test_preds = model.predict(x_test)
        test_metric = metric_fn(y_test, test_preds)

        _log.info(f'Train {metric_fn.__name__}: {train_metric:.3f}')
        _log.info(f'Test {metric_fn.__name__}: {test_metric:.3f}')

    id_to_class = {v: k for k, v in class2label.items()}
    joblib.dump({'model': model,
                 'id_to_class': id_to_class},
                osp.join('output', f'classifier-{checkpoint}.joblib'))


ex.run_commandline()
