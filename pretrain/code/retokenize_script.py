import json
import argparse
import pandas as pd

from tqdm import tqdm
from typing import Dict, List
from transformers import AutoTokenizer


def convert_spans(text: str, word_tokens: List[str], head_data: Dict, tail_data: Dict, tokenizer, relation_type: str):
    """
    Convert word-based spans to subword token-based spans
    """
    # Join tokens to get original text segments
    head_text = head_data['name']
    tail_text = tail_data['name']

    # Get full text before head and tail to calculate offsets
    text_before_head = " ".join(word_tokens[:head_data["pos"][0]])
    text_before_tail = " ".join(word_tokens[:tail_data["pos"][0]])

    # Tokenize full text
    full_encoding = tokenizer(text, add_special_tokens=False)
    tokens = full_encoding.tokens()

    # Find head span in subword tokens
    head_encoding = tokenizer(head_text, add_special_tokens=False)
    head_tokens = head_encoding.tokens()
    head_start = len(tokenizer(text_before_head, add_special_tokens=False).tokens())
    head_end = head_start + len(head_tokens)

    # Find tail span in subword tokens
    tail_encoding = tokenizer(tail_text, add_special_tokens=False)
    tail_tokens = tail_encoding.tokens()
    tail_start = len(tokenizer(text_before_tail, add_special_tokens=False).tokens())
    tail_end = tail_start + len(tail_tokens)

    return {
        "token": word_tokens,
        "h": {
            "name": head_data['name'],
            "pos": [head_start, head_end],
            "type": head_data['type']
        },
        "t": {
            "name": tail_data['name'],
            "pos": [tail_start, tail_end],
            "type": tail_data['type']
        },
        "relation": relation_type
    }


def process_file(input_file, output_file, model_name):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    converted_data = []
    # Read and process data
    with open(input_file, 'r', encoding='utf8') as file:
        for line_number, line in tqdm(enumerate(file, 1)):
            try:
                item = json.loads(line.strip())
                text = " ".join(item["token"])
                converted = convert_spans(
                    text=text,
                    word_tokens=item["token"],
                    head_data=item['h'],
                    tail_data=item['t'],
                    tokenizer=tokenizer,
                    relation_type=item['relation']
                )
                converted_data.append(converted)
            except json.JSONDecodeError as e:
                print(f"Error on line {line_number}: {e}")

    # Save processed data
    pd.DataFrame(converted_data).to_json(output_file, orient='records', lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert word-based spans to subword token-based spans')
    parser.add_argument('--input_file', help='Input JSON file path')
    parser.add_argument('--output_file', help='Output JSON file path')
    parser.add_argument('--model', default='neuralmind/bert-base-portuguese-cased',
                        help='HuggingFace model name for tokenizer')

    args = parser.parse_args()
    process_file(args.input_file, args.output_file, args.model)
