#!/usr/bin/env python
# coding: utf-8


import json

import pandas as pd

import random

from tqdm import tqdm
from pathlib import Path
from itertools import combinations
from typing import List, Dict, Set, Optional
from retokenize_script import process_file

from datalawyer.jurimetria.utils.utils import get_or_create_path
from datalawyer.jurimetria.utils.log_utils import log

default_seed: int = 13370
default_version = '0.24-doutorado'
datalawyer_base_path = Path(f'/media/pedro/repositorios/entidades/dataset/datalawyer/relacao/versao_{default_version}')
datalawyer_pretrain_data_path = datalawyer_base_path / 'fold-0'
datalawyer_recorn_base_path = Path(
    f'/media/pedro/repositorios/RE-Context-or-Names/finetune/supervisedRE/data/datalawyer/v{default_version}')
datalawyer_recorn_cpdata_path = Path(
    f'/media/pedro/repositorios/RE-Context-or-Names/pretrain/data/CP/datalawyer/v{default_version}')

datalawyer_recorn_base_path = get_or_create_path(datalawyer_recorn_base_path)
datalawyer_recorn_cpdata_path = get_or_create_path(datalawyer_recorn_cpdata_path)

# datalawyer_train_path = datalawyer_base_path / 'datalawyer_spert_train_all.json'
# datalawyer_test_path = datalawyer_base_path / 'datalawyer_spert_test_all.json'
# datalawyer_recorn_train_path = datalawyer_recorn_base_path / 'train.txt'
# datalawyer_recorn_test_path = datalawyer_recorn_base_path / 'test.txt'


valid_relations_mappings = {
    'PESSOA': ['FUNCAO'],
    'ORGANIZACAO': ['FUNCAO'],
    'PEDIDO': ['ATRIBUICAO', 'DECISAO', 'VALOR_PEDIDO'],
    'REFLEXO': ['ATRIBUICAO', 'DECISAO', 'VALOR_PEDIDO']
}


def is_relation_valid(entity_1: Dict, entity_2: Dict) -> bool:
    if entity_1['type'] in valid_relations_mappings.keys():
        return entity_2['type'] in valid_relations_mappings[entity_1['type']]
    return False


def convert_to_recorn_item(item: Dict) -> List[Dict]:
    items = []
    tokens = item['tokens']
    entities = item['entities']
    for relation in item['relations']:
        head_entity = entities[relation['head']]
        tail_entity = entities[relation['tail']]
        to_item = {
            'token': tokens,
            'h': {
                'name': ' '.join(tokens[head_entity['start']:head_entity['end'] + 1]),
                'pos': [head_entity['start'], head_entity['end'] + 1],
                'type': head_entity['type']
            },
            't': {
                'name': ' '.join(tokens[tail_entity['start']:tail_entity['end'] + 1]),
                'pos': [tail_entity['start'], tail_entity['end'] + 1],
                'type': tail_entity['type']
            },
            'relation': relation['type']
        }
        items.append(to_item)
    return items


def parse_entity_to_cpdata(data_item: Dict, entity: Dict) -> Dict:
    tokens_positions = list(range(entity['start'], entity['end'] + 1))
    first_position = tokens_positions[0]
    last_position = tokens_positions[-1]
    assert data_item['tokens'][first_position:last_position + 1] == entity['text']
    return {
        'name': ' '.join(entity['text']),
        'pos': [tokens_positions]
    }


def parse_relation_to_cpdata(data_item: Dict) -> List[Dict]:
    items = []
    entities = data_item['entities']
    relations = data_item['relations']
    for relation in relations:
        head_entity = parse_entity_to_cpdata(data_item, entities[relation['head']])
        tail_entity = parse_entity_to_cpdata(data_item, entities[relation['tail']])
        items.append({
            'tokens': data_item['tokens'],
            'h': head_entity,
            'r': relation['type'],
            't': tail_entity
        })
    return items


def matches_entities(entity_1: Dict, entity_2: Dict, head_entity: Dict, tail_entity: Dict):
    head_matches_e1 = entity_1['id'] == head_entity['id']
    head_matches_e2 = entity_2['id'] == head_entity['id']
    tail_matches_e1 = entity_1['id'] == tail_entity['id']
    tail_matches_e2 = entity_2['id'] == tail_entity['id']
    return (head_matches_e1 and tail_matches_e2) or (tail_matches_e1 and head_matches_e2)


def get_relation_for_entities_pair(entity_1: Dict, entity_2: Dict,
                                   relations: List[Dict], entities: List[Dict]) -> Optional[Dict]:
    for relation in relations:
        if matches_entities(entity_1, entity_2, entities[relation['head']], entities[relation['tail']]):
            return relation
    return None


def create_negative_samples(item: Dict, max_negative_samples: int, seed: int = default_seed) -> List[Dict]:
    random.seed(seed)

    items = []
    tokens = item['tokens']
    entities = item['entities']
    relations = item['relations']

    negative_tuples = [(entity_1, entity_2) for entity_1, entity_2 in combinations(entities, 2)
                       if get_relation_for_entities_pair(entity_1, entity_2, relations, entities) is None]

    negative_valid_tuples = [negative_tuple for negative_tuple in negative_tuples if
                             is_relation_valid(negative_tuple[0], negative_tuple[1])]

    samples_size = min(len(negative_valid_tuples), max_negative_samples)
    negative_valid_tuples = random.sample(negative_valid_tuples, samples_size)

    for idx, (head_entity, tail_entity) in enumerate(negative_valid_tuples):
        items.append({
            'token': tokens,
            'h': {
                'name': ' '.join(tokens[head_entity['start']:head_entity['end'] + 1]),
                'pos': [head_entity['start'], head_entity['end'] + 1],
                'type': head_entity['type']
            },
            't': {
                'name': ' '.join(tokens[tail_entity['start']:tail_entity['end'] + 1]),
                'pos': [tail_entity['start'], tail_entity['end'] + 1],
                'type': tail_entity['type']
            },
            'relation': 'no_relation'
        })

    return items


def read_jsonl_with_error_handling(file_path: Path) -> List[Dict]:
    data = []
    with file_path.open('r', encoding='utf8') as file:
        for line_number, line in enumerate(file, 1):
            try:
                json_obj = json.loads(line.strip())
                data.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"Error on line {line_number}: {e}")
    return data


def load_data_items(data_path: Path, relation_types: Set[str], max_negative_samples: int) -> List[Dict]:
    data_items = read_jsonl_with_error_handling(data_path)
    recorn_items = []
    for item in tqdm(data_items, 'Loading items from %s' % str(data_path)):
        if len(item['relations']) > 0:
            for recorn_item in convert_to_recorn_item(item):
                relation_types.add(recorn_item['relation'])
                recorn_items.append(recorn_item)
        for recorn_item in create_negative_samples(item, max_negative_samples):
            relation_types.add(recorn_item['relation'])
            recorn_items.append(recorn_item)
    return recorn_items


def dump_recorn_items(max_negative_samples: int = 10, folds: int = 5,
                      model_name: str = 'neuralmind/bert-base-portuguese-cased', convert_to_hf_tokenizer: bool = False):
    relation_types = set()
    for fold in range(folds):
        for set_type in ['train', 'dev', 'test']:
            datalawyer_data_path = datalawyer_base_path / f'fold-{fold}' / f'{set_type}.json'
            recorn_data_path = get_or_create_path(datalawyer_recorn_base_path / f'fold-{fold}') / f'{set_type}.txt'
            recorn_json_data_path = datalawyer_recorn_base_path / f'fold-{fold}' / f'{set_type}.json'
            data_items = load_data_items(datalawyer_data_path, relation_types, max_negative_samples)
            pd.DataFrame(data_items).to_json(recorn_data_path, orient='records', lines=True)
            log(f'Done saving {len(data_items)} items to {set_type}, fold {fold}, version {default_version}')
            assert recorn_data_path.exists()
            if convert_to_hf_tokenizer:
                process_file(
                    input_file=str(recorn_data_path),
                    output_file=str(recorn_json_data_path),
                    model_name=model_name
                )
                log(f'Done converting entities coordinates from {len(data_items)} items to {set_type}, fold {fold}, '
                    f'version {default_version}')

        rel2id_path = datalawyer_recorn_base_path / f'fold-{fold}' / 'rel2id.json'
        rel_types_list = ['no_relation'] + [relation_type for relation_type in relation_types if
                                            relation_type != 'no_relation']
        json.dump(
            {relation_type: idx for idx, relation_type in enumerate(rel_types_list)},
            rel2id_path.open(mode='w', encoding='utf8')
        )
        assert rel2id_path.exists()


def save_recorn_cpdata():
    cpdata_items = []
    recorn_data_path = datalawyer_recorn_cpdata_path / 'cpdata.json'
    for set_type in ['train', 'dev']:
        data_items = read_jsonl_with_error_handling(datalawyer_pretrain_data_path / f'{set_type}.json')
        for data_item in tqdm(data_items, f'Loading items from {datalawyer_pretrain_data_path / f"{set_type}.json"}'):
            cpdata_items.extend(parse_relation_to_cpdata(data_item))
    print('Saving %d items for pretraining' % len(cpdata_items))
    json.dump(cpdata_items, recorn_data_path.open(mode='w', encoding='utf8'))
    log(f'Done saving {len(cpdata_items)} items for pretraining, version {default_version}')
    assert recorn_data_path.exists()


# In[11]:


# dump_recorn_items()

# In[30]:


save_recorn_cpdata()
