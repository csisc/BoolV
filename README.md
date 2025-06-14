# BoolV
A method to evaluate the response of lightweight LLMs to TRUE-FALSE questions

**Source Code for the Data Visualization:** https://github.com/csisc/BoolV-Analysis.

**To Cite the Work:** Turki, H., Dossou, B. F. P., Nebli, A., & Valdelli, I. (2025). Evaluating the Behavior of Small Language Models in Answering Binary Questions. In *3rd International Workshop on Generalizing from Limited Resources in the Open World (GLOW@IJCAI 2025)*.

# Models
| Model         | Hyperparameters |
| ------------- | --------------- |
| [llama-3.2-1b-instruct-q8_0](https://huggingface.co/lmstudio-community/Llama-3.2-1B-Instruct-GGUF/blob/main/Llama-3.2-1B-Instruct-Q8_0.gguf) | 1.24 B |
| [llama-3.2-3b-instruct-q8_0](https://huggingface.co/lmstudio-community/Llama-3.2-3B-Instruct-GGUF/blob/main/Llama-3.2-3B-Instruct-Q8_0.gguf) | 3.21 B |
| [Phi-3.5-mini-instruct.Q8_0](https://huggingface.co/MaziyarPanahi/Phi-3.5-mini-instruct-GGUF/blob/main/Phi-3.5-mini-instruct.Q8_0.gguf) | 3.82 B |
| [Mistral-7B-Instruct-v0.3.Q8_0](https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF/blob/main/Mistral-7B-Instruct-v0.3.Q8_0.gguf) | 7.25 B |
| [llama-3.2-8b-instruct-q8_0](https://huggingface.co/mradermacher/Llama-3.2-8B-Instruct-GGUF/blob/main/Llama-3.2-8B-Instruct.Q8_0.gguf) | 8.03 B |

# Dataset
- https://github.com/google-research-datasets/boolean-questions
- Train dataset: 9427 labeled training examples.
- Dev dataset: 3270 labeled dev examples.

# Dependencies
- llama-cpp-python
- pathlib
- pandas
- math
- jsonlines

# Funding
This research work has been done thanks to the [computer resources](https://wikimedia.ch/fr/news/swiss-server-helps-optimise-wikidata-in-the-field-of-medicine/) of [Wikimedia Switzerland](https://wikimedia.ch/).
