from llm_sdk import Small_LLM_Model as model
import numpy as np


m = model()

ids = m.encode("What is the sum of 2 and 3?")
ids = np.array(ids[0], dtype=int).tolist()
print(ids[0])
# print(ids.shape)
