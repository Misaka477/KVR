import transformers
print(f"transformers=={transformers.__version__}")

from transformers import AutoConfig, LlamaConfig

model_id = "D:/Software_Development/Project/models/Llama-3.2-1B"

c = AutoConfig.from_pretrained(model_id)

print(f"\nModel: {model_id}")
print(f"  num_hidden_layers: {c.num_hidden_layers}")
print(f"  num_attention_heads: {c.num_attention_heads}")
print(f"  num_key_value_heads: {c.num_key_value_heads}")
print(f"  hidden_size: {c.hidden_size}")
print(f"  intermediate_size: {c.intermediate_size}")
print(f"  vocab_size: {c.vocab_size}")

if hasattr(c, "head_dim"):
    print(f"  head_dim: {c.head_dim}")
else:
    print(f"  d_head (computed): {c.hidden_size // c.num_attention_heads}")

print(f"  rope_theta: {c.rope_theta if hasattr(c, 'rope_theta') else 'N/A'}")
print(f"  max_position_embeddings: {c.max_position_embeddings}")
print(f"  model_type: {c.model_type}")

# Check GQA ratio
n_q = c.num_attention_heads
n_kv = c.num_key_value_heads
print(f"\n  GQA ratio: {n_q}/{n_kv} = {n_q/n_kv:.1f}x")
print(f"  KV head memory vs Q head: {n_kv/n_q*100:.0f}%")