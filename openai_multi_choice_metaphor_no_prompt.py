""" Solving Metaphor Detection via Prompting """
import json
import ast
import logging
import os
from typing import List
import lmppl
from datasets import load_dataset

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
dataset_list = [  # dataset, dataset_name, split
    ['Joanne/Metaphors_and_Analogies', "Quadruples_Green_set", "test"],
    ['Joanne/Metaphors_and_Analogies', 'Pairs_Cardillo_set', "test"],
    ['Joanne/Metaphors_and_Analogies', 'Pairs_Jankowiac_set', "test"],
]


def template_four_words(four_words: List):
    assert len(four_words) == 4, len(four_words)
    return f'{four_words[0]} is to {four_words[1]} what {four_words[2]} is to {four_words[3]}'


def template_sentence(sentence: str):
    return sentence


def get_ppl(model, data, data_name, data_split):
    scorer = lmppl.OpenAI(OPENAI_API_KEY, model=model)
    dataset = load_dataset(data, data_name, split=data_split)
    if data_name == "Quadruples_Green_set":
        dataset_prompt = [[template_four_words(ast.literal_eval(i['stem']) + c) for c in i['pairs']] for i in dataset]
    elif data_name in ["Pairs_Cardillo_set", "Pairs_Jankowiac_set"]:
        dataset_prompt = [[template_sentence(s) for s in i['sentences']] for i in dataset]
    else:
        raise ValueError(f"unknown dataset {data_name}")

    # prompt data
    dataset_index, dataset_flat = [], []
    for n, i in enumerate(dataset_prompt):
        dataset_flat += i
        dataset_index += [n] * len(i)

    # get scores
    scores = {"answer": dataset['answer'], "labels": dataset['labels']}
    ppls = scorer.get_perplexity(input_texts=dataset_flat)
    scores["perplexity"] = [{"input": x, "output": "", "score": float(p), "index": ind} for x, p, ind in zip(dataset_flat, ppls, dataset_index)]
    return scores


if __name__ == '__main__':
    os.makedirs('metaphor_results/scores_openai_no_prompt', exist_ok=True)

    # compute perplexity
    for target_model in ['davinci', 'curie', 'babbage', 'ada']:
        for target_data, target_data_name, target_split in dataset_list:
            scores_file = f"metaphor_results/scores_openai_no_prompt/{target_model}.{os.path.basename(target_data)}_{target_data_name}_{target_split}.json"
            if not os.path.exists(scores_file):
                logging.info(f"[COMPUTING PERPLEXITY] model: `{target_model}`, data: `{target_data}/{target_data_name}/{target_split}`")
                scores_dict = get_ppl(target_model, target_data, target_data_name, target_split)
                with open(scores_file, 'w') as f:
                    json.dump(scores_dict, f)

