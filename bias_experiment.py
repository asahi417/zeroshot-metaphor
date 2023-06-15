""" Solving Metaphor Detection via Prompting """
import json
import logging
import os
import gc
import torch
import lmppl


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
samples = [
    ["This", "is a metaphor."],
    ["This", "is literal."],
    ["This", "is an anomaly."],
    ["This", "is difficult to interpret."],
    ["This sentence", "is a metaphor."],
    ["This sentence", "is literal."],
    ["This sentence", "is an anomaly."],
    ["This sentence", "is difficult to interpret."],
    ["This entire sentence", "is a metaphor."],
    ["This entire sentence", "is literal."],
    ["This entire sentence", "is an anomaly."],
    ["This entire sentence", "is difficult to interpret."],
    ["The sentence that he is referring to", "is a metaphor."],
    ["The sentence that he is referring", "is literal."],
    ["The sentence that he is referring", "is an anomaly."],
    ["The sentence that he is referring", "is difficult to interpret."],
    ["The sentence that he was referring to in our last meeting about the future direction to take", "is a metaphor."],
    ["The sentence that he was referring to in our last meeting about the future direction to take", "is literal."],
    ["The sentence he was referring to in our last meeting about the future direction to take", "is an anomaly."],
    ["The sentence he was referring to in our last meeting about the future direction to take", "is difficult to interpret."],
    ["The sentence that he was referring to in our last meeting about the future direction to take in this long and important project", "is a metaphor."],
    ["The sentence that he was referring to in our last meeting about the future direction to take in this long and important project", "is literal."],
    ["The sentence that he was referring to in our last meeting about the future direction to take in this long and important project", "is an anomaly."],
    ["The sentence that he was referring to in our last meeting about the future direction to take in this long and important project", "is difficult to interpret."],
    ["This thing", "is a metaphor."],
    ["This thing", "is literal."],
    ["This thing", "is an anomaly."],
    ["This thing", "is difficult to interpret."],
    ["This entire thing", "is a metaphor."],
    ["This entire thing", "is literal."],
    ["This entire thing", "is an anomaly."],
    ["This entire thing", "is difficult to interpret."],
    ["The thing that he is referring to", "is a metaphor."],
    ["The thing that he is referring", "is literal."],
    ["The thing that he is referring", "is an anomaly."],
    ["The thing that he is referring", "is difficult to interpret."],
    ["The thing that he was referring to in our last meeting about the future direction to take", "is a metaphor."],
    ["The thing that he was referring to in our last meeting about the future direction to take", "is literal."],
    ["The thing he was referring to in our last meeting about the future direction to take", "is an anomaly."],
    ["The thing he was referring to in our last meeting about the future direction to take", "is difficult to interpret."],
    ["The thing that he was referring to in our last meeting about the future direction to take in this long and important project", "is a metaphor."],
    ["The thing that he was referring to in our last meeting about the future direction to take in this long and important project", "is literal."],
    ["The thing that he was referring to in our last meeting about the future direction to take in this long and important project", "is an anomaly."],
    ["The thing that he was referring to in our last meeting about the future direction to take in this long and important project", "is difficult to interpret."]
]

language_models = {
    # "facebook/opt-30b": [lmppl.LM, 1],  # 30B
    # "facebook/opt-13b": [lmppl.LM, 1],  # 1.3B
    # "facebook/opt-1.3b": [lmppl.LM, 4],  # 1.3B
    # "facebook/opt-350m": [lmppl.LM, 128],   # 350M
    # "facebook/opt-125m": [lmppl.LM, 256],  # 125M
    # "bert-large-cased": [lmppl.MaskedLM, 256],  # 355M
    # "bert-base-cased": [lmppl.MaskedLM, 256],  # 110M
    # "roberta-large": [lmppl.MaskedLM, 128],  # 355M
    # "roberta-base": [lmppl.MaskedLM, 128],  # 110M
    # "google/flan-ul2": [lmppl.EncoderDecoderLM, 1],  # 20B
    # "google/flan-t5-xxl": [lmppl.EncoderDecoderLM, 1],  # 11B
    # "google/flan-t5-xl": [lmppl.EncoderDecoderLM, 4],  # 3B
    # "google/flan-t5-large": [lmppl.EncoderDecoderLM, 128],  # 770M
    # "google/flan-t5-base": [lmppl.EncoderDecoderLM, 256],  # 220M
    # "google/flan-t5-small": [lmppl.EncoderDecoderLM, 256],  # 60M
    "davinci": [lmppl.OpenAI, None]
}

def get_ppl(scoring_model, batch_size):
    # dataset setup
    encoder_decoder = type(scoring_model) is lmppl.EncoderDecoderLM
    # get scores
    if encoder_decoder:
        ppls = scoring_model.get_perplexity(
            input_texts=[x[0] for x in samples],
            output_texts=[x[1] for x in samples],
            batch=batch_size)
        return [{"ppl": y, "input": x_0, "output": x_1} for (x_0, x_1), y in zip(samples, ppls)]
    else:
        ppls = scoring_model.get_perplexity([f"{x_0} {x_1}" for (x_0, x_1) in samples], batch=batch_size)
        return [{"ppl": y, "input": f"{x_0} {x_1}", "output": ""} for (x_0, x_1), y in zip(samples, ppls)]


if __name__ == '__main__':
    os.makedirs('metaphor_results/bias_experiment', exist_ok=True)
    # compute perplexity
    for target_model in language_models.keys():
        scorer = None
        lm_class, batch = language_models[target_model]

        scores_file = f"metaphor_results/bias_experiment/{os.path.basename(target_model)}.json"
        if not os.path.exists(scores_file):
            logging.info(f"[COMPUTING PERPLEXITY] model: `{target_model}`")
            if scorer is None:
                if lm_class is lmppl.MaskedLM:
                    scorer = lm_class(target_model, max_length=256)
                elif lm_class is lmppl.OpenAI:
                    scorer = lm_class(model=target_model, api_key=os.environ['OPENAI_API_KEY'])
                else:
                    scorer = lm_class(target_model, device_map='auto', low_cpu_mem_usage=True, offload_folder=f'offload_folder/{os.path.basename(target_model)}')
            scores_dict = get_ppl(scorer, batch)
            with open(scores_file, 'w') as f:
                json.dump(scores_dict, f)

        del scorer
        gc.collect()
        torch.cuda.empty_cache()

