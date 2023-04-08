""" Solving Metaphor Detection via Prompting
https://docs.google.com/document/d/1ZITlADHkJTuDBlIANCjTr1g5yhiLf7BoGkQyNrzfp4M/edit
"""
import json
import ast
import logging
import os
from time import sleep
from typing import List
import openai
from datasets import load_dataset

openai.api_key = os.getenv("OPENAI_API_KEY", None)
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
label_template = {"metaphor": "is a metaphor", "literal": "is literal", "anomaly": "is difficult to interpret"}
dataset_list = [  # dataset, dataset_name, split
    ['Joanne/Metaphors_and_Analogies', "Quadruples_Green_set", "test"],
    ['Joanne/Metaphors_and_Analogies', 'Pairs_Cardillo_set', "test"],
    ['Joanne/Metaphors_and_Analogies', 'Pairs_Jankowiac_set', "test"],
]


def get_reply(model, text, temperature: float = None):
    while True:
        try:
            if temperature is None:
                print([{"role": "user", "content": text}], model)
                reply = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": text}])
            else:
                reply = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": text}], temperature=temperature)
            break
        except Exception:
            print('Rate limit exceeded. Waiting for 10 seconds.')
            sleep(10)
    return reply['choices'][0]['message']['content']


def prompt(options: List, is_sentence: bool = False, prompt_type="1", cot: bool = False):
    if not is_sentence:
        assert all(len(i) == 4 for i in options), options
        statement = '\n'.join([f'{n+1}) {i[0]} is to {i[1]} what {i[2]} is to {i[3]}' for n, i in enumerate(options)])
    else:
        statement = '\n'.join([f'{n+1}) {i}' for n, i in enumerate(options)])
    if prompt_type == "3":
        __p = ["I will give you three sentences and I would like you to tell me which one is anomalous, which one is "
               "literal, and which one is a metaphor. There is exactly one anomalous sentence, one metaphor, and one "
              f"literal sentence among the three provided sentences. Here are the three sentences:\n{statement}"]
        if cot:
            return [f"{__p[0]}\n Let's think step by step."]
        return __p

    def header(label_type='metaphor'):
        if prompt_type == "1":
            return f"""Answer the question by choosing the correct option.
Which of the following {label_template[label_type]}?"""
        elif prompt_type == "2":
            return f"I will give you three sentences and I would like you to tell me which one {label_template[label_type]}. Here are the three sentences:"
        else:
            ValueError(f"unknown prompt type {prompt_type}")
    if cot:
        text_input_m = f"{header('metaphor')}\n{statement}\nLet's think step by step."
        text_input_l = f"{header('literal')}\n{statement}\nLet's think step by step."
        text_input_a = f"{header('anomaly')}\n{statement}\nLet's think step by step."
    else:
        text_input_m = f"{header('metaphor')}\n{statement}\nThe answer is "
        text_input_l = f"{header('literal')}\n{statement}\nThe answer is "
        text_input_a = f"{header('anomaly')}\n{statement}\nThe answer is "
    return [text_input_m, text_input_l, text_input_a]


def get_chat(model, data, data_name, data_split, prompt_id, cot):
    dataset = load_dataset(data, data_name, split=data_split)
    if data_name == "Quadruples_Green_set":
        dataset_prompt = [prompt([ast.literal_eval(i['stem']) + c for c in i['pairs']], prompt_type=prompt_id, cot=cot) for i in dataset]
    elif data_name in ["Pairs_Cardillo_set", "Pairs_Jankowiac_set"]:
        dataset_prompt = [prompt(i['sentences'], is_sentence=True, prompt_type=prompt_id, cot=cot) for i in dataset]
    else:
        raise ValueError(f"unknown dataset {data_name}")

    # get scores
    scores = {"answer": dataset['answer'], "labels": dataset['labels']}
    if prompt_id == "3":
        output_list = []
        for i in dataset_prompt:
            output_list.append(get_reply(model, i[0]))
        scores["mixed"] = [{"input": x[0], "output": p} for x, p in zip(dataset_prompt, output_list)]
    else:
        for _i, _type in zip([0, 1, 2], ["metaphor", "literal", "anomaly"]):
            output_list = []
            for i in dataset_prompt:
                output_list.append(get_reply(model, i[_i]))
            scores[_type] = [{"input": x[_i], "output": p} for x, p in zip(dataset_prompt, output_list)]
    return scores


if __name__ == '__main__':
    os.makedirs('metaphor_results/chat', exist_ok=True)
    os.makedirs('metaphor_results/chat_cot', exist_ok=True)

    # compute perplexity
    for target_model in ['gpt-3.5-turbo']: #, 'gpt-4']:
        for target_data, target_data_name, target_split in dataset_list:
            for _prompt in ["1", "2", "3"]:
                for trial in [1, 2, 3, 4, 5]:

                    scores_file = f"metaphor_results/chat/{target_model}.{os.path.basename(target_data)}_{target_data_name}_{target_split}.{_prompt}.{trial}.json"
                    if not os.path.exists(scores_file):
                        logging.info(f"[COMPUTING PERPLEXITY] model: `{target_model}`, data: `{target_data}/{target_data_name}/{target_split}`")
                        scores_dict = get_chat(target_model, target_data, target_data_name, target_split, prompt_id=_prompt, cot=False)
                        with open(scores_file, 'w') as f:
                            json.dump(scores_dict, f)

                    scores_file = f"metaphor_results/chat_cot/{target_model}.{os.path.basename(target_data)}_{target_data_name}_{target_split}.{_prompt}.{trial}.json"
                    if not os.path.exists(scores_file):
                        logging.info(f"[COMPUTING PERPLEXITY (COT)] model: `{target_model}`, data: `{target_data}/{target_data_name}/{target_split}`")
                        scores_dict = get_chat(target_model, target_data, target_data_name, target_split, prompt_id=_prompt, cot=True)
                        with open(scores_file, 'w') as f:
                            json.dump(scores_dict, f)

