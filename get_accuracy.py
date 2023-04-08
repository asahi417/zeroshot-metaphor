import json
import os
from statistics import mean
from random import shuffle, seed
from glob import glob
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


model_size = {
    "bert-large-cased": [335, "BERT"],
    "bert-base-cased": [110, "BERT"],
    "roberta-large": [335, "RoBERTa"],
    "roberta-base": [110, "RoBERTa"],
    "google/flan-t5-small": [60, "Flan-T5"],
    "t5-small": [60, "T5"],
    "gpt2": [124, "GPT-2"],
    'facebook/opt-125m': [125, "OPT"],
    "google/flan-t5-base": [220, "Flan-T5"],
    "t5-base": [220, "T5"],
    "facebook/opt-350m": [350, "OPT"],
    "gpt2-medium": [355, "GPT-2"],
    "google/flan-t5-large": [770, "Flan-T5"],
    "t5-large": [770, "T5"],
    "gpt2-large": [774, "GPT-2"],
    "facebook/opt-1.3b": [1300, "OPT"],
    "facebook/opt-iml-1.3b": [1300, "OPT-IML"],
    "facebook/opt-iml-max-1.3b": [1300, "OPT-IML (MAX)"],
    "gpt2-xl": [1500, "GPT-2"],
    "google/flan-t5-xl": [3000, "Flan-T5"],
    "t5-3b": [3000, "T5"],
    "google/flan-t5-xxl": [11000, "Flan-T5"],
    "t5-11b": [11000, "T5"],
    "google/flan-ul2": [20000, "Flan-UL2"],
    "google/ul2": [20000, "UL2"],
    "EleutherAI/gpt-neo-125M": [125, "GPT-J"],
    "EleutherAI/gpt-neo-1.3B": [1300, "GPT-J"],
    "EleutherAI/gpt-neo-2.7B": [2700, "GPT-J"],
    "EleutherAI/gpt-j-6B": [6000, "GPT-J"],
    "EleutherAI/gpt-neox-20b": [20000, "GPT-J"],
    "facebook/opt-30b": [30000, "OPT"],
    "facebook/opt-iml-30b": [30000, "OPT-IML"],
    "facebook/opt-iml-max-30b": [30000, "OPT-IML (MAX)"],
    "facebook/galactica-30b": [30000, "Galactica"],  # 30B
    "facebook/galactica-6.7b": [6700, "Galactica"],  # 6.7B
    "facebook/galactica-1.3b": [1300, "Galactica"],  # 1.3B
    "facebook/galactica-125m": [125, "Galactica"],  # 125
}
model_size = {os.path.basename(k): v for k, v in model_size.items()}


full_accuracy = []
for i in glob("metaphor_results/scores*/*.json"):
    prompt_type = os.path.basename(os.path.dirname(i))
    data_name = os.path.basename(i).split('.')[-2]
    model_name = ".".join(os.path.basename(i).split('.')[:-2])
    if "openai" in prompt_type:
        model_name = f"GPT-3 ({model_name})"

    with open(i, "r") as f:
        data = json.load(f)

    num_option = len(data['labels'][0])
    if "no_prompt" in i:
        ppl_score = pd.DataFrame([i['score'].values for _, i in pd.DataFrame(data['perplexity']).groupby("index")], columns=list(range(num_option))).T
        accuracy_metaphor = []
        accuracy_literal = []
        accuracy_anomaly = []
        for n, label in enumerate(data['labels']):
            if num_option == 2:
                accuracy_metaphor.append(int(data["answer"][n] == ppl_score[n].values.tolist().index(max(ppl_score[n].values.tolist()))))
            else:
                accuracy_literal.append(int(data["answer"][n] == ppl_score[n].values.tolist().index(min(ppl_score[n].values.tolist()))))
                accuracy_anomaly.append(int(data["answer"][n] == ppl_score[n].values.tolist().index(max(ppl_score[n].values.tolist()))))
                accuracy_metaphor.append(int(data["answer"][n] == [i for i in [0, 1, 2] if i not in [accuracy_literal[-1], accuracy_anomaly[-1]]][0]))
    else:
        metaphor_score = pd.DataFrame([i['score'].values for _, i in pd.DataFrame(data['metaphor']).groupby("index")],
                                      columns=list(range(num_option))).T
        literal_score = pd.DataFrame([i['score'].values for _, i in pd.DataFrame(data['literal']).groupby("index")],
                                     columns=list(range(num_option))).T
        anomaly_score = pd.DataFrame([i['score'].values for _, i in pd.DataFrame(data['anomaly']).groupby("index")],
                                     columns=list(range(num_option))).T
        accuracy_metaphor = []
        accuracy_literal = []
        accuracy_anomaly = []
        for n, label in enumerate(data['labels']):
            if num_option == 2:
                accuracy_metaphor.append(int(data["answer"][n] == metaphor_score[n].values.tolist().index(min(metaphor_score[n].values.tolist()))))
            else:
                accuracy_metaphor.append(int(data["answer"][n] == metaphor_score[n].values.tolist().index(min(metaphor_score[n].values.tolist()))))
                accuracy_literal.append(int(data["answer"][n] == literal_score[n].values.tolist().index(min(literal_score[n].values.tolist()))))
                accuracy_anomaly.append(int(data["answer"][n] == anomaly_score[n].values.tolist().index(min(anomaly_score[n].values.tolist()))))
    full_accuracy.append({"Data": data_name, "Label Type": "metaphor", "Model": model_name, "Accuracy": mean(accuracy_metaphor) * 100, "Prompt Type": prompt_type})
    if len(accuracy_literal) != 0:
        full_accuracy.append({"Data": data_name, "Label Type": "literal", "Model": model_name, "Accuracy": mean(accuracy_literal) * 100, "Prompt Type": prompt_type})
    if len(accuracy_anomaly) != 0:
        full_accuracy.append({"Data": data_name, "Label Type": "anomaly", "Model": model_name, "Accuracy": mean(accuracy_anomaly) * 100, "Prompt Type": prompt_type})

df = pd.DataFrame(full_accuracy)
df['Model Size'] = [model_size[i][0] if i in model_size else 0 for i in df['Model']]
df['Model'] = [model_size[i][1] if i in model_size else i for i in df['Model']]
df.to_csv("metaphor_results/result.csv", index=False)

model_order = ["BERT", "RoBERTa", "GPT-2", "GPT-J", "Galactica", "OPT", "T5", "UL2", "OPT-IML", "OPT-IML (MAX)", "Flan-T5", "Flan-UL2"]


def plot(path_to_save: str,
         legend_out: bool = False,
         target_dataset: str = "Metaphors_and_Analogies_Pairs_Jankowiac_set_test",
         target_label: str = "metaphor",
         instruction: bool = False,
         no_prompt: bool = False):
    df_target = df[df['Data'] == target_dataset]
    df_target = df_target[df_target['Label Type'] == target_label]
    if instruction:
        df_target_openai = df_target[df_target['Prompt Type'] == "scores_openai_instruction"]
        df_target = df_target[df_target['Prompt Type'] == "scores_instruction"]
    elif no_prompt:
        df_target_openai = df_target[df_target['Prompt Type'] == "scores_openai_no_prompt"]
        df_target = df_target[df_target['Prompt Type'] == "scores_no_prompt"]
    else:
        df_target_openai = df_target[df_target['Prompt Type'] == "scores_openai"]
        df_target = df_target[df_target['Prompt Type'] == "scores"]

    styles = ['o-', 's-', '^-', "x-", 'o--', 's--', '^--', "x--", 'o:', 's:', '^:', "x:"]
    styles_mark = ['P', "*", "X"]
    styles_lines = ["-", "--", ":", "-."]
    colors = list(mpl.colormaps['tab20b'].colors)
    seed(1)
    shuffle(colors)

    out = df_target.pivot_table(index='Model Size', columns='Model', aggfunc='mean')
    out.columns = [_i[1] for _i in out.columns]
    out = out.reset_index()
    _model_order = [_i for _i in model_order if i in out.columns]

    ax = None
    for _, g in df_target_openai.sort_values(by="Model").iterrows():
        df_tmp = pd.DataFrame([{"Model Size": df_target['Model Size'].min(), "Accuracy": g['Accuracy']},
                               {"Model Size": df_target['Model Size'].max(), "Accuracy": g['Accuracy']}])
        ax = df_tmp.plot.line(ax=ax, y='Accuracy', x='Model Size', color=colors.pop(0), style=styles_lines.pop(0), label=g['Model'], logx=True)

    for _n, c in enumerate(model_order):
        if c not in out.columns:
            continue
        tmp = out[['Model Size', c]].dropna().reset_index()
        tmp['Accuracy'] = tmp[c]
        if len(tmp) == 1:
            tmp.plot.line(y='Accuracy', x='Model Size', ax=ax, color=colors.pop(0), style=styles_mark.pop(0), label=c, logx=True)
        else:
            tmp.plot.line(y='Accuracy', x='Model Size', ax=ax, color=colors.pop(0), style=styles.pop(0), label=c, logx=True)
    if legend_out:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(path_to_save, bbox_inches="tight", dpi=600)


os.makedirs("metaphor_results/figure/prompt", exist_ok=True)
os.makedirs("metaphor_results/figure/no_prompt", exist_ok=True)
os.makedirs("metaphor_results/figure/instruction", exist_ok=True)
for d in ['Metaphors_and_Analogies_Pairs_Jankowiac_set_test', 'Metaphors_and_Analogies_Pairs_Cardillo_set_test', 'Metaphors_and_Analogies_Quadruples_Green_set_test']:
    plot(f"metaphor_results/figure/prompt/plot.{d}.metaphor.png", legend_out=True, target_dataset=d, target_label="metaphor")
    plot(f"metaphor_results/figure/instruction/plot.{d}.instruction.metaphor.png", target_dataset=d, target_label="metaphor", instruction=True)
    plot(f"metaphor_results/figure/no_prompt/plot.{d}.no_prompt.metaphor.png", legend_out=True, target_dataset=d, target_label="metaphor", no_prompt=True)

for d in ['Metaphors_and_Analogies_Pairs_Jankowiac_set_test', 'Metaphors_and_Analogies_Quadruples_Green_set_test']:
    plot(f"metaphor_results/figure/prompt/plot.{d}.literal.png", legend_out=True, target_dataset=d, target_label="literal")
    plot(f"metaphor_results/figure/instruction/plot.{d}.instruction.literal.png", target_dataset=d, target_label="literal", instruction=True)
    plot(f"metaphor_results/figure/no_prompt/plot.{d}.no_prompt.literal.png", legend_out=True, target_dataset=d, target_label="literal", no_prompt=True)
    plot(f"metaphor_results/figure/prompt/plot.{d}.anomaly.png", legend_out=True, target_dataset=d, target_label="anomaly")
    plot(f"metaphor_results/figure/instruction/plot.{d}.instruction.anomaly.png", target_dataset=d, target_label="anomaly", instruction=True)
    plot(f"metaphor_results/figure/no_prompt/plot.{d}.no_prompt.anomaly.png", legend_out=True, target_dataset=d, target_label="anomaly", no_prompt=True)

