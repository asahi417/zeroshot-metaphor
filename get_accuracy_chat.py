import json
import os
from statistics import mean
from random import shuffle, seed
from glob import glob
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


full_accuracy = []
for i in glob("metaphor_results/chat*/*.json"):
    is_cot = "cot" in i
    filename = os.path.basename(i)
    target_data, prompt, trial = filename.split('.')[-4:-1]
    model = "ChatGPT" if "gpt-3.5" in filename else "GPT 4"
    with open(i, "r") as f:
        data = json.load(f)
    data['labels'] = [[_i + 1 for _i in i] for i in data['labels']]
    data['answer'] = [i + 1 for i in data['answer']]

    accuracy = {"trial": trial, "data": target_data, "prompt": prompt, "cot": is_cot}
    if prompt != "3":
        for label, label_id in zip(['metaphor', 'literal', 'anomaly'], [3, 2, 1]):
            if target_data == 'Metaphors_and_Analogies_Pairs_Cardillo_set_test' and label in ['literal', 'anomaly']:
                continue
            tmp_accuracy = []
            for n, d in enumerate(data[label]):
                true = data['labels'][n].index(label_id) + 1
                target = [c for c in d['output'] if c in ['1', '2', '3']]
                if len(target) > 0:
                    tmp_accuracy.append(int(int(target[0]) == true))
                else:
                    # print(f"invalid output: {d['output']}")
                    tmp_accuracy.append(None)
            accuracy[f"{label} (num of invalid)"] = len([i for i in tmp_accuracy if i is None])
            accuracy[label] = mean([i for i in tmp_accuracy if i is not None]) * 100
    else:
        full_prediction = []
        invalid_output = 0
        for n, d in enumerate(data['mixed']):
            # parse output
            try:
                candidates = [i.replace(" ", "")[2:] for i in d['input'].split("\n")[1:]]
                output_list = d['output'].split("\n")
                if len(output_list) != 3:
                    output_list = d['output'].split("\n\n")
                if len(output_list) != 3:
                    prediction = [-1] * 3
                    prediction[int([i for i in d['output'].lower().split("the metaphorical sentence is ")][1][0]) - 1] = 3
                    prediction[int([i for i in d['output'].lower().split("the literal sentence is ")][1][0]) - 1] = 2
                    prediction[int([i for i in d['output'].lower().split("the anomalous sentence is ")][1][0]) - 1] = 1
                else:
                    prediction = []
                    for _d in output_list:
                        if len(_d.split(":")) != 2:
                            if "meta" in _d.lower():
                                label_id = 3  #"metaphor"
                            elif "lite" in _d.lower():
                                label_id = 2  #"litaral"
                            elif "anom" in _d.lower():
                                label_id = 1  #"anomaly"
                            else:
                                assert False

                            cand = [c for c in _d if c in ['1', '2', '3', '4']]
                            assert len(cand) > 0
                            prediction.append([label_id, int(cand[0])])
                        else:
                            label, sentence = _d.split(":")
                            cand = [m for m, c in enumerate(candidates) if c in sentence.lower().replace(" ", "").replace(".", "").replace('"', "")]
                            if len(cand) == 0:
                                cand = [c for c in sentence if c in ['1', '2', '3', '4']]
                                assert len(cand) > 0
                                cand = [int(cand[0])]
                            cand = cand[0]
                            if "meta" in label.lower():
                                label_id = 3  #"metaphor"
                            elif "lite" in label.lower():
                                label_id = 2  #"litaral"
                            elif "anom" in label.lower():
                                label_id = 1  #"anomaly"
                            else:
                                assert False
                            prediction.append([label_id, cand])
                    prediction = [i for i, _ in sorted(prediction, key=lambda x: x[1])]
                full_prediction.append(prediction)

            except Exception:
                print(f"invalid output: {d['output']}")
                # input()
                full_prediction.append(None)
                invalid_output += 1
        for label, label_id in zip(['metaphor', 'literal', 'anomaly'], [3, 2, 1]):
            if target_data == 'Metaphors_and_Analogies_Pairs_Cardillo_set_test' and label in ['literal', 'anomaly']:
                continue
            accuracy[f"{label} (num of invalid)"] = invalid_output
            accuracy[label] = mean([int(p[l.index(label_id)] == label_id) for p, l in zip(full_prediction, data['labels']) if p is not None]) * 100

    full_accuracy.append(accuracy)
df = pd.DataFrame(full_accuracy)
df.to_csv("metaphor_results/result_chat.csv", index=False)
df = df[df["metaphor (num of invalid)"] == 0]
for data, g in df.groupby(["data", "prompt", "cot"]):
    print(data, g.shape)
    print(g.sort_values(by=['metaphor'], ascending=False).head(5)[['prompt', 'cot', 'trial', 'metaphor']])
    print()

