from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import logging


class Gate(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 gate_activation=nn.functional.sigmoid):

        super(Gate, self).__init__()
        self.output_size = output_size

        self.gate_activation = gate_activation
        self.g = nn.Linear(input_size, output_size)
        self.g1 = nn.Linear(output_size, output_size, bias=False)
        self.g2 = nn.Linear(input_size-output_size, output_size, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, x_ent, x_lit):
        x = torch.cat([x_ent, x_lit], 1)
        g_embedded = F.tanh(self.g(x))
        gate = self.gate_activation(self.g1(x_ent) + self.g2(x_lit) + self.gate_bias)
        output = (1-gate) * x_ent + gate * g_embedded

        return output


class LinkPrediction(nn.Module):
    """A general link prediction model with a lookup table for relation
    embeddings."""

    def score_fn(self, heads, tails, rels, eval=False):
        if self.rel_model == 'transe':
            self.normalize_embs = True
            return self.transe_score(heads, tails, rels, eval)
        elif self.rel_model == 'distmult':
            return self.distmult_score(heads, tails, rels, eval)
        elif self.rel_model == 'complex':
            return self.complex_score(heads, tails, rels, eval)
        elif self.rel_model == 'simple':
            return self.simple_score(heads, tails, rels, eval)
        elif self.rel_model == 'ermlp':
            return self.ermlp_score(heads, tails, rels, eval)
        else:
            raise ValueError(f'Unknown relational model {self.rel_model}.')

    def __init__(self, dim, rel_model, loss_fn, num_relations, regularizer,
                 num_neighbors=5, edge_features=False, weighted_pooling=False):

        super().__init__()

        self.dim = dim
        self.normalize_embs = False
        self.regularizer = regularizer
        self.rel_model = rel_model
        self.num_neighbors = num_neighbors
        self.weighted_pooling = weighted_pooling
        self.edge_features = edge_features
        self.logger = logging.getLogger()
        self.num_relations = num_relations

        if edge_features:
            self.logger.info('Loading relation features from file.')

            self.rel_emb = nn.Sequential(OrderedDict([
                ('emb', nn.Embedding(num_relations, 768)),
                ('linear', nn.Linear(768, self.dim))
            ]))

            relation_desc_features = torch.load('data/Wikidata5M/relation_description_embedding.pt')
            self.rel_emb.emb.weight.data = relation_desc_features
        else:
            self.rel_emb = nn.Embedding(num_relations, self.dim)
            nn.init.xavier_uniform_(self.rel_emb.weight.data)

        if self.rel_model == 'ermlp':
            self.ermlp_linear_hidden = nn.Linear(self.dim * 3, self.dim)
            self.ermlp_linear_out = nn.Linear(self.dim, 1)

        self.neighborhood_pooling = torch.nn.AvgPool2d((self.num_neighbors, 1), stride=1)

        if loss_fn == 'margin':
            self.loss_fn = margin_loss
        elif loss_fn == 'nll':
            self.loss_fn = nll_loss
        else:
            raise ValueError(f'Unknown loss function {loss_fn}')

    def encode(self, *args, **kwargs):
        ent_emb = self._encode_entity(*args, **kwargs)
        if self.normalize_embs:
            ent_emb = F.normalize(ent_emb, dim=-1)

        return ent_emb

    def _encode_entity(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def compute_loss(self, ent_embs, rels, neg_idx):
        """
        ent_embs torch.Size([512, 2, 128])
        rels torch.Size([512, 1])
        neg_idx torch.Size([512, 64, 2])
        """
        batch_size = ent_embs.shape[0]

        # Scores for positive samples
        rels = self.rel_emb(rels)
        heads, tails = torch.chunk(ent_embs, chunks=2, dim=1)
        # pos heads torch.Size([512, 1, 128])
        pos_scores = self.score_fn(heads, tails, rels)

        if self.regularizer > 0:
            reg_loss = self.regularizer * l2_regularization(heads, tails, rels)
        else:
            reg_loss = 0

        # Scores for negative samples
        neg_embs = ent_embs.view(batch_size * 2, -1)[neg_idx]
        heads, tails = torch.chunk(neg_embs, chunks=2, dim=2)
        # neg heads torch.Size([512, 64, 1, 128])

        neg_scores = self.score_fn(heads.squeeze(), tails.squeeze(), rels)

        # pos vs neg size: torch.Size([512, 1]) torch.Size([512, 64])
        model_loss = self.loss_fn(pos_scores, neg_scores)
        return model_loss + reg_loss

    def transe_score(self, heads, tails, rels, eval=False):
        return -torch.norm(heads + rels - tails, dim=-1, p=1)

    def distmult_score(self, heads, tails, rels, eval=False):
        return torch.sum(heads * rels * tails, dim=-1)

    def complex_score(self, heads, tails, rels, eval=False):
        heads_re, heads_im = torch.chunk(heads, chunks=2, dim=-1)
        tails_re, tails_im = torch.chunk(tails, chunks=2, dim=-1)
        rels_re, rels_im = torch.chunk(rels, chunks=2, dim=-1)

        return torch.sum(rels_re * heads_re * tails_re +
                         rels_re * heads_im * tails_im +
                         rels_im * heads_re * tails_im -
                         rels_im * heads_im * tails_re,
                         dim=-1)

    def simple_score(self, heads, tails, rels, eval=False):
        heads_h, heads_t = torch.chunk(heads, chunks=2, dim=-1)
        tails_h, tails_t = torch.chunk(tails, chunks=2, dim=-1)
        rel_a, rel_b = torch.chunk(rels, chunks=2, dim=-1)

        return torch.sum(heads_h * rel_a * tails_t +
                         tails_h * rel_b * heads_t, dim=-1) / 2

    def ermlp_score(self, heads, tails, rels, eval=False):

        batch_size = heads.size(0)
        num_samples = heads.size(1)
        if not eval:
            rels = rels.repeat(1, heads.size(1), 1)
        linear_input = torch.cat([heads.reshape((-1, self.dim)),
                                  rels.reshape((-1, self.dim)),
                                  tails.reshape((-1, self.dim))], dim=-1)
        linear_hidden = self.ermlp_linear_hidden(linear_input)
        linear_hidden = torch.tanh(linear_hidden)
        linear_out = self.ermlp_linear_out(linear_hidden)
        #linear_out = torch.sigmoid(linear_out)
        linear_out = linear_out.reshape((batch_size, num_samples))
        return linear_out




def print_tensor_stats(name, tensor):
    # print(f'{name} size: {tensor.size()}, nan: {torch.isnan(tensor).any()}, inf: {torch.isinf(tensor).any()}', tensor)
    pass



class InductiveLinkPrediction(LinkPrediction):
    """Description-based Link Prediction (DLP)."""

    def _encode_entity(self, text_tok, text_mask):
        raise NotImplementedError

    def weighted_neighborhood_pooling(self, features):
        """
        features torch.Size([batch_size * num_neighbors, dim])
        """
        num_samples = features.size(0)
        weights = torch.arange(self.num_neighbors, 0, -1).repeat(self.dim, 1).T.repeat(num_samples, 1).reshape(-1, self.num_neighbors, self.dim).to(features.device)
        avg = torch.sum(weights * features, dim=1) / torch.sum(weights, dim=1)
        return avg

    def forward(self, text_tok, text_mask, rels=None, neg_idx=None, text_tok_neighborhood_all=None):
        """
        Shapes:
            text_tok torch.Size([batch_size * 2, max_len])
            text_tok_neighborhood_all torch.Size([batch_size * 2, num_neighbors, max_len])
        """

        batch_size, num_text_tokens = text_tok.size(0), text_tok.size(-1)

        if text_tok_neighborhood_all is None:
            # without neighborhood
            # Encode text into an entity representation from its description
            ent_embs = self.encode(text_tok.view(-1, num_text_tokens),
                                   text_mask.view(-1, num_text_tokens))
        else:
            # with neighborhood
            num_text_tokens_neigh = text_tok_neighborhood_all.size(-1)

            # flatten neighbors in order to be able to insert separation token inbetween them
            # text_tok_neighborhood_fused torch.Size([batch_size, 2 * num_neighbors, max_len])
            text_tok_neighborhood_all = text_tok_neighborhood_all.view(batch_size, -1, num_text_tokens_neigh)

            # uses tokenizer.sep_token_id = 102 for separating neighbors,
            # added between neighbors torch.Size([batch_size, 2 * num_neighbors, max_len + 1])
            text_tok_neighborhood_fused = torch.cat([text_tok_neighborhood_all,
                                         torch.full((text_tok_neighborhood_all.size(0),
                                                     text_tok_neighborhood_all.size(1), 1), 102).to(text_tok_neighborhood_all.device)],
                                        dim=2)

           # text_tok + text_tok_neighborhood_fused
            # entities
            num_entities = text_tok.size(0) if neg_idx is None else text_tok.size(0) * 2

            ent_feature = torch.cat([text_tok.view(-1, num_text_tokens),
                                     torch.full((num_entities, 1), 102).to(text_tok.device),
                                     text_tok_neighborhood_fused.view(-1, self.num_neighbors * num_text_tokens_neigh + self.num_neighbors)], dim=1)

            ent_feature_mask = (ent_feature > 0).float()

            ent_embs = self.encode(ent_feature,
                                   ent_feature_mask)

        if neg_idx is None:
            # Forward is being used to compute entity embeddings only
            out = ent_embs
        else:
            # Forward is being used to compute link prediction loss
            ent_embs = ent_embs.view(batch_size, 2, -1)
            out = self.compute_loss(ent_embs, rels, neg_idx)

        return out


class NeighborhoodSelectionTransformer(nn.Module):

    def __init__(self, encoder_name):
        super().__init__()
        self.encoder_neigh_att = BertModel.from_pretrained(encoder_name,
                                                           output_attentions=False,
                                                           output_hidden_states=False)

        self.enc_neigh_att_linear = nn.Linear(self.encoder_neigh_att.config.hidden_size, 1)

    def forward(self, text_tok, text_mask):
        x = self.encoder_neigh_att(text_tok, text_mask)[0][:, 0]
        x = self.enc_neigh_att_linear(x)
        x = torch.sigmoid(x)
        return x


class BertEmbeddingsLP(InductiveLinkPrediction):
    """BERT for Link Prediction (BLP)."""

    # todo rework parameters and remove out-dated ones
    def __init__(self, dim, rel_model, loss_fn, num_relations, encoder_name,
                 regularizer, num_neighbors=5, edge_features=False, weighted_pooling=False):
        super().__init__(dim, rel_model, loss_fn, num_relations, regularizer, num_neighbors,
                        edge_features, weighted_pooling)
        self.encoder = BertModel.from_pretrained(encoder_name,
                                                 output_attentions=False,
                                                 output_hidden_states=False)
        hidden_size = self.encoder.config.hidden_size
        self.enc_linear = nn.Linear(hidden_size, self.dim, bias=False)

    def _encode_entity(self, text_tok, text_mask):
        # Extract BERT representation of [CLS] token
        embs = self.encoder(text_tok, text_mask)[0][:, 0]
        embs = self.enc_linear(embs)
        return embs



class WordEmbeddingsLP(InductiveLinkPrediction):
    """Description encoder with pretrained embeddings, obtained from BERT or a
    specified tensor file.
    """

    def __init__(self, rel_model, loss_fn, num_relations, regularizer,
                 dim=None, encoder_name=None, embeddings=None, num_neighbors=5, edge_features=False,
                 weighted_pooling=False):
        if not encoder_name and not embeddings:
            raise ValueError('Must provided one of encoder_name or embeddings')

        if encoder_name is not None:
            encoder = BertModel.from_pretrained(encoder_name)
            embeddings = encoder.embeddings.word_embeddings
            embedding_dim = embeddings.embedding_dim
        else:
            emb_tensor = torch.load(embeddings)
            num_embeddings, embedding_dim = emb_tensor.shape
            embeddings = nn.Embedding(num_embeddings, embedding_dim)
            embeddings.weight.data = emb_tensor

        if dim is None:
            dim = embeddings.embedding_dim

        super().__init__(dim, rel_model, loss_fn, num_relations, regularizer, num_neighbors,
                         edge_features, weighted_pooling)

        self.embeddings = embeddings

    def _encode_entity(self, text_tok, text_mask):
        raise NotImplementedError


class BOW(WordEmbeddingsLP):
    """Bag-of-words (BOW) description encoder, with BERT low-level embeddings.
    """

    def _encode_entity(self, text_tok, text_mask=None):
        if text_mask is None:
            text_mask = torch.ones_like(text_tok, dtype=torch.float)
        # Extract average of word embeddings

        embs = self.embeddings(text_tok)
        lengths = torch.sum(text_mask, dim=-1, keepdim=True)
        embs = torch.sum(text_mask.unsqueeze(dim=-1) * embs, dim=1)
        embs = embs / lengths

        return embs


class DKRL(WordEmbeddingsLP):
    """Description-Embodied Knowledge Representation Learning (DKRL) with CNN
    encoder, after
    Zuo, Yukun, et al. "Representation learning of knowledge graphs with
    entity attributes and multimedia descriptions."
    """

    def __init__(self, dim, rel_model, loss_fn, num_relations, regularizer,
                 encoder_name=None, embeddings=None, num_neighbors=5, edge_features=False,
                 weighted_pooling=False):
        super().__init__(rel_model, loss_fn, num_relations, regularizer,
                         dim, encoder_name, embeddings,
                         num_neighbors, edge_features, weighted_pooling)

        emb_dim = self.embeddings.embedding_dim
        self.conv1 = nn.Conv1d(emb_dim, self.dim, kernel_size=2)
        self.conv2 = nn.Conv1d(self.dim, self.dim, kernel_size=2)

    def _encode_entity(self, text_tok, text_mask):
        if text_mask is None:
            text_mask = torch.ones_like(text_tok, dtype=torch.float)
        # Extract word embeddings and mask padding
        embs = self.embeddings(text_tok) * text_mask.unsqueeze(dim=-1)

        # Reshape to (N, C, L)
        embs = embs.transpose(1, 2)
        text_mask = text_mask.unsqueeze(1)

        # Pass through CNN, adding padding for valid convolutions
        # and masking outputs due to padding
        embs = F.pad(embs, [0, 1])
        embs = self.conv1(embs)
        embs = embs * text_mask
        if embs.shape[2] >= 4:
            kernel_size = 4
        elif embs.shape[2] == 1:
            kernel_size = 1
        else:
            kernel_size = 2
        embs = F.max_pool1d(embs, kernel_size=kernel_size)
        text_mask = F.max_pool1d(text_mask, kernel_size=kernel_size)
        embs = torch.tanh(embs)
        embs = F.pad(embs, [0, 1])
        embs = self.conv2(embs)
        lengths = torch.sum(text_mask, dim=-1)
        embs = torch.sum(embs * text_mask, dim=-1) / lengths
        embs = torch.tanh(embs)

        return embs


class TransductiveLinkPrediction(LinkPrediction):
    def __init__(self, dim, rel_model, loss_fn, num_entities, num_relations,
                 regularizer):
        super().__init__(dim, rel_model, loss_fn, num_relations, regularizer)
        self.ent_emb = nn.Embedding(num_entities, dim)
        nn.init.xavier_uniform_(self.ent_emb.weight.data)

    def _encode_entity(self, entities):
        return self.ent_emb(entities)

    def forward(self, pos_pairs, rels, neg_idx):
        embs = self.encode(pos_pairs)
        return self.compute_loss(embs, rels, neg_idx)


def margin_loss(pos_scores, neg_scores):
    loss = 1 - pos_scores + neg_scores
    loss[loss < 0] = 0
    return loss.mean()


def nll_loss(pos_scores, neg_scores):
    return (F.softplus(-pos_scores).mean() + F.softplus(neg_scores).mean()) / 2


def l2_regularization(heads, tails, rels):
    reg_loss = 0.0
    for tensor in (heads, tails, rels):
        reg_loss += torch.mean(tensor ** 2)

    return reg_loss / 3.0
