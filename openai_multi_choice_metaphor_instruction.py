""" Solving Metaphor Detection via Prompting """
import json
import ast
import logging
import os
from typing import List
import lmppl
from datasets import load_dataset
from string import ascii_letters

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
label_template = {"metaphor": "is a metaphor", "literal": "is literal", "anomaly": "is difficult to interpret"}
template_header = "Which of the following examples"
template_footer = "The answer is "


def prompt(options: List, is_sentence: bool = False):
    if not is_sentence:
        assert all(len(i) == 4 for i in options), options
        statement = '\n'.join([f'{ascii_letters[n]}) {i[0]} is to {i[1]} what {i[2]} is to {i[3]}' for n, i in enumerate(options)])
    else:
        statement = '\n'.join([f'{ascii_letters[n]}) {i}' for n, i in enumerate(options)])
    text_input_m = f"{template_header} {label_template['metaphor']}?\n{statement}\n{template_footer}"
    text_input_l = f"{template_header} {label_template['literal']}?\n{statement}\n{template_footer}"
    text_input_a = f"{template_header} {label_template['anomaly']}?\n{statement}\n{template_footer}"
    return [[
        f"{text_input_m}\n{ascii_letters[n]}",
        f"{text_input_l}\n{ascii_letters[n]}",
        f"{text_input_a}\n{ascii_letters[n]}"
    ] for n in range(len(options))]


dataset_list = [  # dataset, dataset_name, split
    ['Joanne/Metaphors_and_Analogies', "Quadruples_Green_set", "test"],
    ['Joanne/Metaphors_and_Analogies', 'Pairs_Cardillo_set', "test"],
    ['Joanne/Metaphors_and_Analogies', 'Pairs_Jankowiac_set', "test"],
]


def get_ppl(model, data, data_name, data_split):
    scorer = lmppl.OpenAI(OPENAI_API_KEY, model=model)
    dataset = load_dataset(data, data_name, split=data_split)
    if data_name == "Quadruples_Green_set":
        dataset_prompt = [prompt([ast.literal_eval(i['stem']) + c for c in i['pairs']]) for i in dataset]
    elif data_name in ["Pairs_Cardillo_set", "Pairs_Jankowiac_set"]:
        dataset_prompt = [prompt(i['sentences'], is_sentence=True) for i in dataset]
    else:
        raise ValueError(f"unknown dataset {data_name}")

    # prompt data
    dataset_index, dataset_flat = [], []
    for n, i in enumerate(dataset_prompt):
        dataset_flat += i
        dataset_index += [n] * len(i)

    # get scores
    scores = {"answer": dataset['answer'], "labels": dataset['labels']}
    for _i, _type in zip([0, 1, 2], ["metaphor", "literal", "anomaly"]):
        _dataset_flat = [i[_i] for i in dataset_flat]
        ppls = scorer.get_perplexity(input_texts=_dataset_flat)
        scores[_type] = [{"input": x, "output": "", "score": float(p), "index": ind} for x, p, ind in zip(_dataset_flat, ppls, dataset_index)]
    return scores


if __name__ == '__main__':
    os.makedirs('metaphor_results/scores_openai_instruction', exist_ok=True)

    # compute perplexity
    for target_model in ['davinci', 'curie', 'babbage', 'ada']:
        for target_data, target_data_name, target_split in dataset_list:
            scores_file = f"metaphor_results/scores_openai_instruction/{target_model}.{os.path.basename(target_data)}_{target_data_name}_{target_split}.json"
            if not os.path.exists(scores_file):
                logging.info(f"[COMPUTING PERPLEXITY] model: `{target_model}`, data: `{target_data}/{target_data_name}/{target_split}`")
                scores_dict = get_ppl(target_model, target_data, target_data_name, target_split)
                with open(scores_file, 'w') as f:
                    json.dump(scores_dict, f)

