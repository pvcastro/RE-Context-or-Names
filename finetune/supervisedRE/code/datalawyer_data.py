#!/usr/bin/env python
# coding: utf-8

# In[10]:


import json

import pandas as pd

import random

from tqdm import tqdm
from pathlib import Path
from itertools import combinations
from typing import List, Dict, Tuple, Set

# In[2]:


random_seed: int = 13370
random.seed(random_seed)
version = 0.16
datalawyer_base_path = Path('/media/discoD/repositorios/entidades/dataset/datalawyer/relacao/versao_{}'.format(version))
datalawyer_recorn_base_path = Path(
    '/media/discoD/repositorios/RE-Context-or-Names/finetune/supervisedRE/data/datalawyer')

# In[3]:


datalawyer_recorn_base_path.mkdir(exist_ok=True)

# In[4]:


datalawyer_train_path = datalawyer_base_path / 'datalawyer_spert_train_all.json'
datalawyer_test_path = datalawyer_base_path / 'datalawyer_spert_test_all.json'
datalawyer_recorn_train_path = datalawyer_recorn_base_path / 'train.txt'
datalawyer_recorn_test_path = datalawyer_recorn_base_path / 'test.txt'

# In[5]:


valid_relations_mappings = {
    'PESSOA': ['FUNCAO'],
    'ORGANIZACAO': ['FUNCAO'],
    'PEDIDO': ['ATRIBUICAO', 'DECISAO', 'VALOR_PEDIDO'],
    'REFLEXO': ['ATRIBUICAO', 'DECISAO', 'VALOR_PEDIDO']
}


# In[8]:


def is_relation_valid(entity_1: Dict, entity_2: Dict) -> bool:
    if entity_1['type'] in valid_relations_mappings.keys():
        return entity_2['type'] in valid_relations_mappings[entity_1['type']]
    elif entity_2['type'] in valid_relations_mappings.keys():
        return entity_1['type'] in valid_relations_mappings[entity_2['type']]
    else:
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
                'name': ' '.join(tokens[head_entity['start']:head_entity['end']]),
                'pos': [head_entity['start'], head_entity['end']],
                'type': head_entity['type']
            },
            't': {
                'name': ' '.join(tokens[tail_entity['start']:tail_entity['end']]),
                'pos': [tail_entity['start'], tail_entity['end']],
                'type': tail_entity['type']
            },
            'relation': relation['type']
        }
        items.append(to_item)
    return items


def matches_entities(entity_1: Dict, entity_2: Dict, head_entity: Dict, tail_entity: Dict):
    head_matches_e1 = entity_1['id'] == head_entity['id']
    head_matches_e2 = entity_2['id'] == head_entity['id']
    tail_matches_e1 = entity_1['id'] == tail_entity['id']
    tail_matches_e2 = entity_2['id'] == tail_entity['id']
    return (head_matches_e1 and tail_matches_e2) or (tail_matches_e1 and head_matches_e2)


def get_relation_for_entities_pair(entity_1: Dict, entity_2: Dict,
                                   relations: List[Dict], entities: List[Dict]) -> Dict:
    for relation in relations:
        if matches_entities(entity_1, entity_2, entities[relation['head']], entities[relation['tail']]):
            return relation
    return None


def get_head_tail_entities(entity_1: Dict, entity_2: Dict) -> Tuple[Dict, Dict]:
    if entity_1['type'] in valid_relations_mappings.keys():
        return entity_1, entity_2
    elif entity_2['type'] in valid_relations_mappings.keys():
        return entity_2, entity_1
    else:
        raise ValueError(
            'Invalid entities for relation:\nEntity 1:{}\nEntity 2:{}'.format(str(entity_1), str(entity_2))
        )


def create_negative_samples(item: Dict, max_negative_samples: int) -> List[Dict]:
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

    for idx, (entity_1, entity_2) in enumerate(negative_valid_tuples):
        head_entity, tail_entity = get_head_tail_entities(entity_1, entity_2)

        items.append({
            'token': tokens,
            'h': {
                'name': ' '.join(tokens[head_entity['start']:head_entity['end']]),
                'pos': [head_entity['start'], head_entity['end']],
                'type': head_entity['type']
            },
            't': {
                'name': ' '.join(tokens[tail_entity['start']:tail_entity['end']]),
                'pos': [tail_entity['start'], tail_entity['end']],
                'type': tail_entity['type']
            },
            'relation': 'no_relation'
        })

    return items


def load_data_items(data_path: Path, relation_types: Set[str], max_negative_samples: int) -> Tuple[
    List[Dict], Set[str]]:
    data_items = json.load(data_path.open(mode='r', encoding='utf8'))
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


def dump_recorn_items(max_negative_samples: int = 10):
    relation_types = set()
    for set_type in ['train', 'test']:
        datalawyer_data_path = datalawyer_base_path / 'datalawyer_spert_{}_all.json'.format(set_type)
        recorn_data_path = datalawyer_recorn_base_path / '{}.txt'.format(set_type)
        data_items = load_data_items(datalawyer_data_path, relation_types, max_negative_samples)
        print('Saving %d items for %s dataset' % (len(data_items), set_type))
        pd.DataFrame(data_items).to_json(recorn_data_path, orient='records', lines=True)

    rel2id_path = datalawyer_recorn_base_path / 'rel2id.json'
    rel_types_list = ['no_relation'] + [relation_type for relation_type in relation_types if
                                        relation_type != 'no_relation']
    json.dump({relation_type: idx for idx, relation_type in enumerate(rel_types_list)},
              rel2id_path.open(mode='w', encoding='utf8'))


# In[11]:


dump_recorn_items()

# In[43]:


train_data = json.load(datalawyer_train_path.open(mode='r', encoding='utf8'))

# In[47]:


count = 0
for data in train_data:
    if data['tokens'] == ['The', 'victims', 'were', 'traveling', 'Monday', 'in', 'a', 'caravan', 'with', 'other',
                          'members', 'of', 'El', 'Salvador', "'s", 'delegation', 'to', 'the', 'Central', 'American',
                          'Parliament', ',', 'a', 'regional', 'body', 'composed', 'of', 'representatives', 'elected',
                          'in', 'the', 'member', 'states', 'of', 'El', 'Salvador', ',', 'Guatemala', ',', 'Honduras',
                          ',', 'Panama', ',', 'Nicaragua', 'and', 'the', 'Dominican', 'Republic', ',', 'Reyes', 'said',
                          '.']:
        print(data)
        break

# In[ ]:
