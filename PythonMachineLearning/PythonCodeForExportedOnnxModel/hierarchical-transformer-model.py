import torch
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
from MongoDa import positive_pairs, negative_pairs

# 1. Load a lightweight, highly efficient base model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Prepare your training triplets: (User Query text, Correct Hierarchical Path, Hard Negative Path)
train_examples = [
    InputExample(texts=[pos['file'], pos['folder'], neg['folder']])
    for pos, neg in zip(positive_pairs, negative_pairs)
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(model)

# 3. Fine-tune the model to understand your specific 9000+ node taxonomy
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3, warmup_steps=100)

# 4. Export the fine-tuned model directly to ONNX format for ML.NET compatibility
# This creates a 'model.onnx' file and tokenization configs
output_dir = "hierarchical_transformer_model"
model.save(output_dir)
#model.save_pretrained('hierarchical_transformer_model')

onnx_model = SentenceTransformer(
    output_dir,
    backend="onnx",
    model_kwargs={"export": True},
)
onnx_model.save(output_dir + "_onnx")