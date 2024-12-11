import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import os
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
import os
from datetime import datetime
import seaborn as sns

warnings.filterwarnings("ignore")


SUPER_WEIGHT = (1, 2533, 7890)
MLP_DIM = 11008
HIDDEN_DIM = 4096

def setup_model():
    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf", 
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="./cache_dir"
    )
    model.eval()
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", cache_dir="./cache_dir")
    tokenizer.pad_token = tokenizer.eos_token  # Add this line
    return model, tokenizer

def plot_super_weight_value(model):
    """
    Plot a histogram of all the values in the 1st layer of the down projection.
    Highlight the value of superweight with a red dot"""
    # Set the style and figure size
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))
    
    layer = model.model.layers[SUPER_WEIGHT[0]]
    layer_down_project = layer.mlp.down_proj
    print(f"Layer down project shape: {layer_down_project.weight.shape}")
    all_values = layer_down_project.weight.detach().cpu().numpy().flatten()
    print(f"Plotting values: {len(all_values)}")
    
    # Plot histogram with log scale
    sns.histplot(all_values, bins=50, color='skyblue', alpha=0.6)
    plt.yscale('log')
    
    # Add superweight point
    index = SUPER_WEIGHT[1] * MLP_DIM + SUPER_WEIGHT[2]
    superweight_value = all_values[index]
    plt.scatter(superweight_value, 1, color='red', s=100, zorder=5, label='Superweight')
    
    # Add labels and title
    plt.xlabel('Weight Value', fontsize=12)
    plt.ylabel('Count (log scale)', fontsize=12)
    plt.title('Distribution of Down-Projection Weights\nLayer {}'.format(SUPER_WEIGHT[0]), 
              fontsize=14, pad=20)
    
    # Add annotation for superweight value
    plt.annotate(f'Superweight: {superweight_value:.4f}',
                xy=(superweight_value, 1), xytext=(10, 10),
                textcoords='offset points', ha='left',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plot_path = Path(__file__).parent / "plots"
    plot_path.mkdir(parents=True, exist_ok=True)
    plot_path = plot_path / f"super_weight_plot_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def get_super_weight_value(model):
    layer = model.model.layers[SUPER_WEIGHT[0]]
    layer_down_project = layer.mlp.down_proj  # size: (11008, 4096) or (MLP_DIM, HIDDEN_DIM)
    return layer_down_project.weight[SUPER_WEIGHT[1], SUPER_WEIGHT[2]].item()

def remove_super_weight(model):
    with torch.no_grad():
        model.model.layers[SUPER_WEIGHT[0]].mlp.down_proj.weight[SUPER_WEIGHT[1], SUPER_WEIGHT[2]] = 0.0
    return model

def generate_text(model, tokenizer, prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def run_experiment(prompt="My favorite condiment is"):
    model, tokenizer = setup_model()

    plot_super_weight_value(model)
    
    print(f"Original super weight value: {get_super_weight_value(model)}")
    
    print("\nGenerating with original model:")
    print(generate_text(model, tokenizer, prompt))
    
    model = remove_super_weight(model)
    print("\nGenerating with super weight removed:")
    print(generate_text(model, tokenizer, prompt))

if __name__ == "__main__":
    run_experiment()