import argparse
import json

import torch
import transformers
import evaluate
from datasets import Dataset
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_scheduler
from transformers import BertTokenizer
from transformers import AutoModelForSequenceClassification

transformers.logging.set_verbosity_error()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="FB15k-237", type=str, help="Dataset to use, choose from: Wikidata5M, FB15k-237")
    parser.add_argument("--epochs", default=10, type=int, help="number of training epochs")
    parser.add_argument("--mode", default="train+predict", type=str, help="train+predict, train, predict")

    args = parser.parse_args()
    dataset_name = args.dataset
    num_epochs = args.epochs
    mode = args.mode

    # -- load data --
    maps = torch.load(f'./data/{dataset_name}/maps.pt')
    uri_to_id = maps['ent_ids']
    relation_uri_to_id = maps['rel_ids']
    id_to_relation_uri = {v: k for k, v in relation_uri_to_id.items()}

    if 'train' in mode:
        edge_index = []
        edge_type = []
        with open(f'./data/{dataset_name}/ind-train.tsv') as triples_in:
            for line in triples_in:
                head, relation, tail = line[:-1].split('\t')
                edge_index.append([uri_to_id[head], uri_to_id[tail]])
                edge_type.append(relation_uri_to_id[relation])

        sentences = {}
        with open(f'./data/{dataset_name}/entity2textlong.txt') as sentences_in:
            for line in sentences_in:
                uri, description = line[:-1].split('\t')[:2]
                if uri not in uri_to_id:
                    continue
                sentences[uri_to_id[uri]] = description

        samples = []
        print('creating samples...')
        for i in tqdm(range(len(edge_index))):
            head_id = edge_index[i][0]
            tail_id = edge_index[i][1]
            if head_id in sentences and tail_id in sentences:
                samples.append((sentences[head_id], sentences[tail_id], edge_type[i]))

        print('num samples:', len(samples))


        # -- tokenize data and create dataloader --
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        dataset = []
        print('tokenizing...')
        for idx, (first_sentence, second_sentence, label) in enumerate(tqdm(samples)):
            encoding = tokenizer(first_sentence, second_sentence, padding="max_length", truncation=True)
            encoding['idx'] = idx
            encoding['labels'] = label
            dataset.append(encoding)
            if idx == 10000:
                break

        d = Dataset.from_list(dataset)
        d.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
        dataloader = torch.utils.data.DataLoader(d, batch_size=64)

        # -- train model --
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(relation_uri_to_id.keys()))
        optimizer = AdamW(model.parameters(), lr=5e-5)

        num_training_steps = num_epochs * len(dataloader)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        torch.cuda.empty_cache()

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        model.train()

        progress_bar = tqdm(range(num_training_steps))

        #model.load_state_dict(
        #    torch.load(f'./data/{dataset_name}/relation_classification_model.pt'))
        for epoch in tqdm(range(num_epochs)):
            loss_total = 0
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                loss_total += loss.detach().cpu()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
            print('loss_total:', loss_total)

        print('saving model...')
        torch.save(model.state_dict(), f'./data/{dataset_name}/relation_classification_model.pt')
        # -- evaluate model --
        print('evaluating...')
        metric = evaluate.load("accuracy")
        model.eval()
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        acc = metric.compute()
        print('acc:', acc)

    if 'predict' in mode:
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(
            relation_uri_to_id.keys()))
        model.load_state_dict(
            torch.load(f'./data/{dataset_name}/relation_classification_model.pt'))

        page_links = json.load(open(f'./data/{dataset_name}/page_links.json'))

        sentences = {}
        with open(f'./data/{dataset_name}/entity2textlong.txt') as sentences_in:
            for line in sentences_in:
                uri, description = line[:-1].split('\t')
                if uri not in uri_to_id:
                    continue
                sentences[uri_to_id[uri]] = description

        # -- tokenize data and create dataloader --
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        model.to('cuda')
        model.eval()

        dataset = []
        batch_edge_index = []
        idx = 0
        triples_idx_tensor = []
        for head, tails in tqdm(page_links.items()):
            for tail in tails:
                if head in uri_to_id and tail in uri_to_id:
                    if uri_to_id[head] in sentences and uri_to_id[tail] in sentences:
                        encoding = tokenizer(sentences[uri_to_id[head]], sentences[uri_to_id[tail]], padding="max_length", truncation=True)
                        encoding['idx'] = idx
                        batch_edge_index.append([idx, head, tail])
                        dataset.append(encoding)
                        idx += 1

                if idx % 64 == 0:
                    d = Dataset.from_list(dataset)
                    d.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask"])
                    dataloader = torch.utils.data.DataLoader(d, batch_size=64, shuffle=False)
                    with torch.no_grad():
                        for batch in dataloader:
                            batch = {k: v.to('cuda') for k, v in batch.items()}
                            with torch.no_grad():
                                outputs = model(**batch)
                                predictions = torch.argmax(outputs.logits, dim=1)

                    with open(f'./data/{dataset_name}/page_link_graph_typed.txt', 'a+') as triples_out:
                        for (_, head, tail), relation in zip(batch_edge_index, predictions.tolist()):
                            triples_out.write(
                                '\t'.join([str(head), str(id_to_relation_uri[relation]), str(tail)]) + '\n')
                            triples_idx_tensor.append([uri_to_id[head], relation, uri_to_id[tail]])

                    dataset = []
                    batch_edge_index = []

        triples_idx_tensor = torch.unique(torch.tensor(triples_idx_tensor), dim=0)
        torch.save(triples_idx_tensor, f'./data/{dataset_name}/page_link_graph_typed.pt')



