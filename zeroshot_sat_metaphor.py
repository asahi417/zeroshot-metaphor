""" Solving Metaphor Detection via Prompting """
import json
import logging
import os
import gc
import torch
import lmppl
from datasets import load_dataset


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
dataset = load_dataset("relbert/analogy_questions_private", "sat")
label_template = {"metaphor": "is a metaphor.", "literal": "is literal.", "anomaly": "is difficult to interpret."}
all_pairs = []
for s in ['test', 'validation']:
    for n, i in enumerate(dataset[s]):
        a, b = i['stem']
        for m, (c, d) in enumerate(i['choice']):
            statement = f'"{a} is to {b} what {c} is to {d}"'
            all_pairs.append({
                "target": [a, b, c, d], "label": i['answer'] == m, "index": n, "statement": statement
            })


language_models = {
    "facebook/opt-iml-30b": [lmppl.LM, 1],  # 30B
    "facebook/opt-iml-max-30b": [lmppl.LM, 1],  # 30B
    "facebook/opt-iml-max-1.3b": [lmppl.LM, 4],  # 1.3B
    "facebook/opt-iml-1.3b": [lmppl.LM, 4],  # 1.3B
    "facebook/opt-30b": [lmppl.LM, 1],  # 30B
    "facebook/opt-13b": [lmppl.LM, 1],  # 1.3B
    "facebook/opt-1.3b": [lmppl.LM, 4],  # 1.3B
    "facebook/opt-350m": [lmppl.LM, 128],  # 350M
    "facebook/opt-125m": [lmppl.LM, 256],  # 125M
    "EleutherAI/gpt-neox-20b": [lmppl.LM, 1],  # 20B
    "EleutherAI/gpt-j-6B": [lmppl.LM, 4],  # 6B
    "EleutherAI/gpt-neo-2.7B": [lmppl.LM, 8],  # 2.7B
    "EleutherAI/gpt-neo-1.3B": [lmppl.LM, 8],  # 1.3B
    "EleutherAI/gpt-neo-125M": [lmppl.LM, 256],  # 125M
    "gpt2-xl": [lmppl.LM, 8],  # 1.5B
    "gpt2-large": [lmppl.LM, 128],  # 774M
    "gpt2-medium": [lmppl.LM, 256],  # 355M
    "gpt2": [lmppl.LM, 512],  # 124M
    "bert-large-cased": [lmppl.MaskedLM, 256],  # 355M
    "bert-base-cased": [lmppl.MaskedLM, 256],  # 110M
    "roberta-large": [lmppl.MaskedLM, 128],  # 355M
    "roberta-base": [lmppl.MaskedLM, 128],  # 110M
    "google/ul2": [lmppl.EncoderDecoderLM, 1],  # 20B
    "t5-11b": [lmppl.EncoderDecoderLM, 1],  # 11B
    "t5-3b": [lmppl.EncoderDecoderLM, 4],  # 3B
    "t5-large": [lmppl.EncoderDecoderLM, 128],  # 770M
    "t5-base": [lmppl.EncoderDecoderLM, 512],  # 220M
    "t5-small": [lmppl.EncoderDecoderLM, 512],  # 60M
    "google/flan-ul2": [lmppl.EncoderDecoderLM, 1],  # 20B
    "google/flan-t5-xxl": [lmppl.EncoderDecoderLM, 1],  # 11B
    "google/flan-t5-xl": [lmppl.EncoderDecoderLM, 4],  # 3B
    "google/flan-t5-large": [lmppl.EncoderDecoderLM, 128],  # 770M
    "google/flan-t5-base": [lmppl.EncoderDecoderLM, 256],  # 220M
    "google/flan-t5-small": [lmppl.EncoderDecoderLM, 256],  # 60M
    "davinci": [lmppl.OpenAI, None]
}

def get_ppl(scoring_model, batch_size, label_siffix):
    # dataset setup
    encoder_decoder = type(scoring_model) is lmppl.EncoderDecoderLM
    # get scores
    if encoder_decoder:
        ppls = scoring_model.get_perplexity(
            input_texts=[x['statement'] for x in all_pairs],
            output_texts=[label_siffix] * len(all_pairs),
            batch=batch_size)
        return [{"ppl": y, "input": x['statement'], "output": label_siffix,
                 "label": x['label'], "index": x['index'], "target": x['target']} for x, y in zip(all_pairs, ppls)]
    else:
        ppls = scoring_model.get_perplexity([f"{x['statement']} {label_siffix}" for x in all_pairs], batch=batch_size)
        return [{"ppl": y, "input": f"{x['statement']} {label_siffix}", "output": "",
                 "label": x['label'], "index": x['index'], "target": x['target']} for x, y in zip(all_pairs, ppls)]


if __name__ == '__main__':
    os.makedirs('metaphor_results/scores_sat', exist_ok=True)
    # compute perplexity
    for target_model in language_models.keys():
        scorer = None
        lm_class, batch = language_models[target_model]
        for label, suffix in label_template.items():
            scores_file = f"metaphor_results/scores_sat/{os.path.basename(target_model)}.{label}.json"
            if not os.path.exists(scores_file):
                if scorer is None:
                    if lm_class is lmppl.MaskedLM:
                        scorer = lm_class(target_model, max_length=256)
                    elif lm_class is lmppl.OpenAI:
                        scorer = lm_class(model=target_model, api_key=os.environ['OPENAI_API_KEY'])
                    else:
                        scorer = lm_class(target_model, device_map='auto', low_cpu_mem_usage=True, offload_folder=f'offload_folder/{os.path.basename(target_model)}')
                logging.info(f"[COMPUTING PERPLEXITY] model: `{target_model}`, label: `{label}`")
                scores_dict = get_ppl(scorer, batch, suffix)
                with open(scores_file, 'w') as f:
                    json.dump(scores_dict, f)

        del scorer
        gc.collect()
        torch.cuda.empty_cache()

