import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# Hyperparameters
TARGET_LAYERS = [0, 1, 2, 3, 4]  # Layers to search for super weights
TOP_K_WEIGHTS = 100  # Number of top weights to identify per layer
PROMPT = "The capital of France is"
CACHE_DIR = "./cache_dir"
MODEL_NAME = "yahma/llama-13b-hf"
COMBINED_SCORE_THRESHOLD = 0.01

# Step 1: Setup model and tokenizer
def setup_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=CACHE_DIR
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token  # Add this line
    return model, tokenizer

# Step 2: Extract top weights by magnitude
def get_top_weights(model, layer_idx, k=TOP_K_WEIGHTS):
    layer = model.model.layers[layer_idx]
    weight_matrix = layer.mlp.down_proj.weight
    
    # Flatten and get top-k weights
    flat_indices = torch.topk(torch.abs(weight_matrix.flatten()), k).indices
    top_weights = []
    for idx in flat_indices:
        row = idx // weight_matrix.size(1)
        col = idx % weight_matrix.size(1)
        weight_value = weight_matrix[row, col].item()
        top_weights.append((row.item(), col.item(), weight_value))
    return top_weights

# Step 3: Capture activations for a given input
def get_activations(model, tokenizer, prompt, layer_idx, weight_indices):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Add position_ids
    position_ids = torch.arange(inputs.input_ids.shape[1], dtype=torch.long, device=model.device)
    position_ids = position_ids.unsqueeze(0)
    
    activations = {}
    with torch.no_grad():
        # Initial embedding
        hidden_state = model.model.embed_tokens(inputs.input_ids)
        
        for i, layer in enumerate(model.model.layers):
            # Forward pass through the layer with position_ids
            layer_outputs = layer(hidden_state, position_ids=position_ids)
            hidden_state = layer_outputs[0]
            
            if i == layer_idx:
                # Get MLP activations
                mlp_output = layer.mlp.up_proj(hidden_state)
                down_proj_output = layer.mlp.down_proj(mlp_output)
                
                for row, col, _ in weight_indices:
                    activations[(row, col)] = down_proj_output[0, :, row].mean().item()
                break
    
    return activations

# Step 4: Combine weights and activations
def analyze_super_weights(model, tokenizer, prompt):
    results = {}
    for layer_idx in TARGET_LAYERS:
        print(f"Analyzing Layer {layer_idx}...")

        # Get top weights by magnitude
        top_weights = get_top_weights(model, layer_idx)

        # Get activations for these weights
        activations = get_activations(model, tokenizer, prompt, layer_idx, top_weights)

        # Combine weights and activations
        combined_results = []
        for (row, col, weight_value) in top_weights:
            activation_value = activations.get((row, col), 0.0)
            combined_results.append({
                "row": row,
                "col": col,
                "weight_value": weight_value,
                "activation_value": activation_value,
                "combined_score": abs(weight_value * activation_value)
            })

        # Sort by combined score
        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
        results[layer_idx] = combined_results

    return results

# Step 5: Main function to run the analysis
def main():
    model, tokenizer = setup_model()
    results = analyze_super_weights(model, tokenizer, PROMPT)

    # Print results
    for layer_idx, layer_results in results.items():
        print(f"\nTop results for Layer {layer_idx}:")
        for result in layer_results[:10]:  # Display top 10 for each layer if combined score is greater than COMBINED_SCORE_THRESHOLD
            if result["combined_score"] > COMBINED_SCORE_THRESHOLD:
                print(result)

if __name__ == "__main__":
    main()
