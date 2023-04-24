
# This document contains prompts to solve metaphor detection with chatgpt/gpt4.

Dataset:
Cardillo: 
"The war campaign was a hard push." -> metaphor
"The drunk's shove was an angry push." -> literal
Jankowiac:
“This scholar is an inventor" -> literal
"This balloon is an inventor" -> anomaly
"Imagination is an inventor" -> metaphor
Green:
Query: ['answer', 'riddle']
Pairs: [[ "jersey", "number" ], [ "key", "lock" ], [ "solution", "problem" ]]
Each pair is transformed into a sentence by template of A is to B what C is to D.

Setting: Three sentences (A, B, and C), where one of them is metaphor, and another one is literal, and the other one is an anomaly.

## Prompt 1:
Answer the question by choosing the correct option.
Which of the following is a metaphor?
1) A
2) B
3) C
The answer is

[Example]
Answer the question by choosing the correct option.
Which of the following is a metaphor?
1) This scholar is an inventor
2) This balloon is an inventor
3) Imagination is an inventor
The answer is

## Prompt 2:
I will give you three sentences and I would like you to tell me which one is a metaphor. Here are the three sentences:
1) A
2) B
3) C
The answer is

[Example]
I will give you three sentences and I would like you to tell me which one is a metaphor. Here are the three sentences:
1) This scholar is an inventor
2) This balloon is an inventor
3) Imagination is an inventor
The answer is

## Prompt 3:
I will give you three sentences and I would like you to tell me which one is anomalous, which one is literal, and which one is a metaphor. There is exactly one anomalous sentence, one metaphor, and one literal sentence among the three provided sentences. Here are the three sentences:
1) A
2) B
3) C
The answer is

[Example]
I will give you three sentences and I would like you to tell me which one is anomalous, which one is literal, and which one is a metaphor. There is exactly one anomalous sentence, one metaphor, and one literal sentence among the three provided sentences. Here are the three sentences:
1) This scholar is an inventor
2) This balloon is an inventor
3) Imagination is an inventor
The answer is




