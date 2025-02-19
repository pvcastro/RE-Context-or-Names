import json
import itertools
from typing import List, Dict
from pathlib import Path


def by_sentence(item: dict) -> str:
    return ' '.join(item['tokens'])


def by_relation_type(item: dict) -> str:
    return item['r']


def filter_list(items: List[Dict]) -> List[Dict]:
    items_set = set()
    items_list = list()
    for item in items:
        item_str = str(item)
        if len(items_set) == 0:
            items_set.add(item_str)
            items_list.append(item)
        else:
            if item_str not in items_set:
                items_set.add(item_str)
                items_list.append(item)
            else:
                print(f'Deduped sentence: {item_str}')
    return items_list


if __name__ == "__main__":
    data = json.load(open("../data/CP/datalawyer/v0.24-doutorado/cpdata.json", encoding='utf8', mode='r'))

    print('Found %d training items' % len(data))

    dedup = False
    if dedup:

        items_by_sentences = {key: list(group) for key, group in
                              itertools.groupby(sorted(data, key=by_sentence), by_sentence)}

        print('Found %d different sentences from the %d training items' % (len(items_by_sentences), len(data)))

        mult_data = {sentence: items_by_sentences[sentence] for sentence in items_by_sentences.keys() if
                     len(items_by_sentences[sentence]) > 1}

        print('Found %d different sentences containing more than 1 training item' % len(mult_data))

        count = 0
        difference = 0
        dedup_items = []
        for sentence in items_by_sentences.keys():
            items = items_by_sentences[sentence]
            filtered_items = filter_list(items)
            dedup_items.extend(filtered_items)
            if len(items) != len(filtered_items):
                count += 1
                print(count)
                difference += (len(items) - len(filtered_items))

        print('Found %d sentences containing a total of %d duplicate training items' % (count, difference))

        items_by_relation = {key: list(group) for key, group in
                             itertools.groupby(sorted(dedup_items, key=by_relation_type), by_relation_type)}

        ll = 0
        rel2scope = {}
        list_data = []
        for key in items_by_relation.keys():
            list_data.extend(items_by_relation[key])
            rel2scope[key] = [ll, len(list_data)]
            ll = len(list_data)

        json.dump(list_data, open("../data/CP/datalawyer/v0.24-doutorado/cpdata_dedup.json", mode="w", encoding='utf8'))
        json.dump(rel2scope, open("../data/CP/datalawyer/v0.24-doutorado/rel2scope.json", mode="w", encoding='utf8'))

        print('Saved deduped dataset with %d items' % len(dedup_items))

    else:

        items_by_relation = {key: list(group) for key, group in
                             itertools.groupby(sorted(data, key=by_relation_type), by_relation_type)}

        ll = 0
        rel2scope = {}
        list_data = []
        for key in items_by_relation.keys():
            list_data.extend(items_by_relation[key])
            rel2scope[key] = [ll, len(list_data)]
            ll = len(list_data)

        json.dump(rel2scope, open("../data/CP/datalawyer/v0.24-doutorado/rel2scope.json", mode="w", encoding='utf8'))

        print('Saved dataset with %d items' % len(data))
