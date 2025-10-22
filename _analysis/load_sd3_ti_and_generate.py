import os
import torch
import safetensors
from diffusers import StableDiffusion3Pipeline
from typing import Optional

# Helper function to get the embedding layer from a text encoder
def get_embedding_layer(text_encoder):
    if hasattr(text_encoder, "get_input_embeddings"):
        return text_encoder.get_input_embeddings()
    elif hasattr(text_encoder, "shared"): # For T5 models
        return text_encoder.shared
    else:
        raise AttributeError(f"Text encoder of type {type(text_encoder)} does not have a known embedding layer attribute.")

def load_pipeline_with_ti_from_local_folder_step(
    local_embeddings_folder: str,
    base_model_name_or_path: str,
    placeholder_token: str, # This is the BASE placeholder token, e.g., "<WhatsApp>"
    training_step: int,
    torch_dtype: torch.dtype = torch.float16,
    device: Optional[str] = None,
) -> StableDiffusion3Pipeline:
    """
    Loads a Stable Diffusion 3 pipeline and MANUALLY injects Textual Inversion embeddings
    from a local folder, for a specific training step.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if not placeholder_token:
        raise ValueError("placeholder_token must be provided.")
    if not isinstance(training_step, int) or training_step <= 0:
        raise ValueError("training_step must be a positive integer.")
    print(f"Manually loading TI for placeholder token: '{placeholder_token}' for training step: {training_step}")

    embedding_files_templates = [
        {"template": "learned_embeds_t1-steps-{}.safetensors", "encoder_attr": "text_encoder", "tokenizer_attr": "tokenizer"},
        {"template": "learned_embeds_t2-steps-{}.safetensors", "encoder_attr": "text_encoder_2", "tokenizer_attr": "tokenizer_2"},
        {"template": "learned_embeds_t3-steps-{}.safetensors", "encoder_attr": "text_encoder_3", "tokenizer_attr": "tokenizer_3"},
    ]

    print(f"Loading base model '{base_model_name_or_path}'...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch_dtype,
        # It's important that the text encoders are loaded with the pipeline here
    )
    print("Base pipeline loaded.")

    for info in embedding_files_templates:
        filename = info["template"].format(training_step)
        file_path = os.path.join(local_embeddings_folder, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Embedding file '{filename}' not found in folder '{local_embeddings_folder}'. "
                f"Expected path: {file_path}"
            )

        print(f"Manually processing TI embeddings from '{file_path}' for {info['encoder_attr']}...")
        
        # Load the saved embeddings tensor
        loaded_embeddings_dict = safetensors.torch.load_file(file_path, device="cpu")
        if placeholder_token not in loaded_embeddings_dict:
            raise ValueError(f"Placeholder token '{placeholder_token}' not found as a key in {filename}. Available keys: {list(loaded_embeddings_dict.keys())}")
        
        learned_embedding_tensor = loaded_embeddings_dict[placeholder_token]
        num_vectors = learned_embedding_tensor.shape[0]
        print(f"  Found {num_vectors} vector(s) for token '{placeholder_token}' in {info['encoder_attr']}.")

        # Prepare all placeholder strings for this encoder
        all_placeholder_strings = [placeholder_token]
        if num_vectors > 1:
            for i in range(1, num_vectors):
                all_placeholder_strings.append(f"{placeholder_token}_{i}")
        
        # Get the specific text_encoder and tokenizer from the pipeline
        text_encoder = getattr(pipe, info["encoder_attr"])
        tokenizer = getattr(pipe, info["tokenizer_attr"])

        # Add new tokens to the tokenizer
        num_added_tokens = tokenizer.add_tokens(all_placeholder_strings)
        if num_added_tokens != len(all_placeholder_strings):
            # This might happen if some tokens were already in the tokenizer, which is unusual for TI placeholders
            print(f"  Warning: Expected to add {len(all_placeholder_strings)} tokens to {info['tokenizer_attr']}, but {num_added_tokens} were newly added.")
            # Check if they already existed and raise if any are missing after trying to add
            current_vocab = tokenizer.get_vocab()
            for tk_str in all_placeholder_strings:
                if tk_str not in current_vocab:
                    # This is problematic if a token we need wasn't successfully added or already present
                    raise RuntimeError(f"Token '{tk_str}' was not found in {info['tokenizer_attr']} after add_tokens call.")

        new_token_ids = tokenizer.convert_tokens_to_ids(all_placeholder_strings)
        if len(new_token_ids) != num_vectors:
            raise RuntimeError(f"Mismatch in token ID count for {info['encoder_attr']}. Expected {num_vectors}, got {len(new_token_ids)}.")

        # Resize the token embeddings in the text encoder
        text_encoder.resize_token_embeddings(len(tokenizer))
        print(f"  Resized {info['encoder_attr']} embeddings to {len(tokenizer)} tokens.")

        # Get the embedding layer of the text_encoder
        embedding_layer = get_embedding_layer(text_encoder)
        
        # Copy the learned embeddings into the text encoder
        with torch.no_grad():
            for i in range(num_vectors):
                token_id_to_update = new_token_ids[i]
                embedding_vector_to_copy = learned_embedding_tensor[i]
                embedding_layer.weight[token_id_to_update] = embedding_vector_to_copy.to(
                    device=embedding_layer.weight.device, 
                    dtype=embedding_layer.weight.dtype
                )
        
        print(f"  Successfully copied {num_vectors} embeddings into {info['encoder_attr']}.")

    print(f"All Textual Inversion embeddings for step {training_step} MANUALLY loaded from local folder.")
    
    pipe.to(device)
    print(f"Pipeline moved to {device}.")

    return pipe

if __name__ == '__main__':
    # --- Configuration ---
    # Path to the folder containing the step-specific embeddings
    local_folder_with_embeddings = "/mnt/lustre/work/oh/owl661/compositional-vaes/sd3_whatsappA_embedding/wandb-a5a0j11k"
    base_model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
    
    # !!! IMPORTANT: You MUST provide the placeholder token that was used during training !!!
    # This is the actual string used in your prompts during training (e.g., "<sks>", "<my-style>", etc.)
    # The training script textual_inversion_sd3.py uses args.placeholder_token
    # Check your training command or W&B config for this value.
    placeholder_token_string = "<whatsappA>" # Replace with your actual token, e.g. "<whatsappA-style>" or whatever you used
    
    training_step_to_load = 1500
    output_image_prefix = "ti_whatsapp_gen"

    # Device and Dtype
    generation_device = "cuda" if torch.cuda.is_available() else "cpu"
    generation_dtype = torch.float16 if generation_device == "cuda" else torch.float32 # float16 for CUDA, float32 for CPU

    if placeholder_token_string == "<YOUR-TRAINING-PLACEHOLDER-TOKEN>" or not placeholder_token_string:
        print("ERROR: Please set the 'placeholder_token_string' variable in the script with the actual placeholder token used during training.")
    else:
        print(f"Attempting to load TI embeddings for placeholder '{placeholder_token_string}' at step {training_step_to_load} from: {local_folder_with_embeddings}")
        
        try:
            pipeline = load_pipeline_with_ti_from_local_folder_step(
                local_embeddings_folder=local_folder_with_embeddings,
                base_model_name_or_path=base_model_id,
                placeholder_token=placeholder_token_string,
                training_step=training_step_to_load,
                torch_dtype=generation_dtype,
                device=generation_device 
            )
            print(f"\nPipeline with Textual Inversion embeddings for step {training_step_to_load} loaded successfully!")

            prompts_to_generate = [
                f"A realistic photo of a {placeholder_token_string} logo on a sleek smartphone.",
                f"Concept art of a futuristic city with buildings shaped like the {placeholder_token_string} icon, digital painting.",
                f"A child's drawing of the {placeholder_token_string} app icon.",
                f"Close-up of a {placeholder_token_string} notification bubble, vibrant colors.",
                f"A sticker of the {placeholder_token_string} logo."
            ]

            num_inference_steps = 28 
            guidance_scale = 0.0

            for i, prompt in enumerate(prompts_to_generate):
                print(f"\nGenerating image {i+1}/{len(prompts_to_generate)} with prompt: '{prompt}'")
                
                # Set seed for reproducibility of generations if desired for testing
                # generator = torch.Generator(device=generation_device).manual_seed(42 + i) 
                generator = None # For random generations

                image = pipeline(
                    prompt, 
                    num_inference_steps=num_inference_steps, 
                    guidance_scale=guidance_scale,
                    generator=generator
                ).images[0]
                
                output_filename = f"{output_image_prefix}_step{training_step_to_load}_prompt{i+1}.png"
                image.save(output_filename)
                print(f"Image saved to ./{output_filename}")

            print("\nAll generations complete.")

        except FileNotFoundError as fnfe:
            print(f"File error: {fnfe}")
        except ValueError as ve:
            print(f"Configuration error: {ve}")
        except AttributeError as ae:
            print(f"Attribute error: {ae}. This might indicate an issue with the diffusers library version or the way methods are called.")
            print("Consider updating diffusers: pip install --upgrade diffusers transformers accelerate")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc() 