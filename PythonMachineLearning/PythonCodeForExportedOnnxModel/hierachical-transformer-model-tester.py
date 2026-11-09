import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# 1) Load the tokenizer matching this model
tokenizer = AutoTokenizer.from_pretrained(
    "D:/Pract/Mine/Misc/TreeNodeSearchModel/hierarchical_transformer_model_onnx"
)

# 2) Text to encode
texts = [
    "Achintya in Bali.png"
]

# 3) Tokenize to model inputs
encoded = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors="np"
)

# 4) Build ONNX session
session = ort.InferenceSession(
    "D:/Pract/Mine/Misc/TreeNodeSearchModel/hierarchical_transformer_model_onnx/onnx/model.onnx",
    providers=["CPUExecutionProvider"]
)

# 5) Run inference
inputs = {
    "input_ids": encoded["input_ids"].astype(np.int64),
    "attention_mask": encoded["attention_mask"].astype(np.int64),
    "token_type_ids": encoded["token_type_ids"].astype(np.int64),
}

outputs = session.run(None, inputs)

# output[0] is the last_hidden_state: [batch_size, seq_len, dim]
token_embeddings = outputs[0] 
attention_mask = encoded["attention_mask"] # [batch_size, seq_len]

# --- Mean Pooling Implementation ---
# Expand mask to match embedding dimensions: [batch_size, seq_len, 1]
mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(float)

# Sum embeddings while ignoring padding tokens
sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)

# Count non-padding tokens for each sequence
sum_mask = np.clip(attention_mask.sum(axis=1, keepdims=True), a_min=1e-9, a_max=None)

# Calculate mean: [batch_size, dim]
embeddings = sum_embeddings / sum_mask


# shape should be: [num_texts, 384]
print(embeddings.shape)   # Now it will be: (1, 384)
print(embeddings)