from llama_cpp import Llama
import pathlib
import pandas as pd
import math
import jsonlines

#Defining models
MODEL_Q8_0 = Llama(
    model_path="Llama-3.2-1B-Instruct-Q8_0.gguf",
    n_ctx=128, n_gpu_layers=128, logits_all=True
)

#Defining function for getting a response
def query_with_logprobs(model, question):
    prompt = f"Q: {question} A:"
    output = model(prompt=prompt, max_tokens=1, temperature=10000, logprobs=True)
    response = output["choices"][0]
    logprobs = response["logprobs"]["top_logprobs"][0]  # Get logprobs for first token

    # Extract logprobs for TRUE
    logprob_true = math.exp(logprobs.get(" TRUE", float("-inf")))
    return logprob_true

#Defining dataframe
df = pd.DataFrame(columns=['question', 'answer', 'probability'])

# Path to your JSONL file
file_path = 'train.jsonl'

# Open and read the JSONL file
with jsonlines.open(file_path) as reader:
    for obj in reader:
        df = df._append({'question': obj["question"], 'answer': obj["answer"], 'probability': query_with_logprobs(MODEL_Q8_0, obj["question"]+"? Answer with one word: FALSE or TRUE.")}, ignore_index=True)
        print(df.index)

# Saving dataframe
df.to_excel("benchmark_evaluation_false_first.xlsx", index=False)
