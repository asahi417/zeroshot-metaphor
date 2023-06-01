# Zero-shot Metaphor Solver based on Perplexity

Following scripts are all to reproduce the zero-shot metaphor with LMs.

- Public Models (HuggingFace)
  - [zeroshot_multi_choice_metaphor.py](zeroshot_multi_choice_metaphor.py): Basic prompt (`"A is to B what C is to D" is a metaphor.`). 
  - [zeroshot_multi_choice_metaphor_instruction.py](zeroshot_multi_choice_metaphor_instruction.py): Prompt with all the options and asking the answer (QA style).
  - [zeroshot_multi_choice_metaphor_no_prompt.py](zeroshot_multi_choice_metaphor_no_prompt.py): Basic prompt without the label suffix (`A is to B what C is to D.`).
- Private Model (GPT-3 from OpenAI)
  - [openai_multi_choice_metaphor.py](openai_multi_choice_metaphor.py): Basic prompt (`"A is to B what C is to D" is a metaphor.`).
  - [openai_multi_choice_metaphor_chat.py](openai_multi_choice_metaphor_chat.py): Chat models such as ChatGPT & GPT-4.
  - [openai_multi_choice_metaphor_instruction.py](openai_multi_choice_metaphor_instruction.py): Prompt with all the options and asking the answer (QA style).
  - [openai_multi_choice_metaphor_no_prompt.py](openai_multi_choice_metaphor_no_prompt.py): Basic prompt without the label suffix (`A is to B what C is to D.`).

Misc:
- [zeroshot_sat_metaphor.py](zeroshot_sat_metaphor.py): Basic prompt for SAT dataset (both of public/private models).