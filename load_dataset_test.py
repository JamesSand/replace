

from datasets import load_dataset

hf_dataset = "HuggingFaceH4/math-500"
ds = load_dataset(hf_dataset, split="test")

print(len(ds))



