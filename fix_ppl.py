import json
from glob import glob
from math import exp, log


# for _file in glob("metaphor_results/scores_openai_no_prompt/*.json"):
#     print(_file)
#     with open(_file) as f:
#         tmp = json.load(f)
#         perplexity = tmp['perplexity']
#         new = []
#         for i in perplexity:
#             i['score'] = exp(-log(i['score']))
#         tmp['perplexity'] = perplexity
#     with open(_file, "w") as f:
#         json.dump(tmp, f)


# for _file in glob("metaphor_results/scores_openai/*.json"):
#     print(_file)
#     with open(_file) as f:
#         tmp = json.load(f)
#         for _type in ['metaphor', 'literal', 'anomaly']:
#             perplexity = tmp[_type]
#             new = []
#             for i in perplexity:
#                 i['score'] = exp(-log(i['score']))
#             tmp[_type] = perplexity
#     with open(_file, "w") as f:
#         json.dump(tmp, f)



for _file in glob("metaphor_results/scores_openai_instruction/*.json"):
    print(_file)
    with open(_file) as f:
        tmp = json.load(f)
        for _type in ['metaphor', 'literal', 'anomaly']:
            perplexity = tmp[_type]
            new = []
            for i in perplexity:
                i['score'] = exp(-log(i['score']))
            tmp[_type] = perplexity
    with open(_file, "w") as f:
        json.dump(tmp, f)