import os
import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_name):
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        padding_side='left'  # Fix the padding warning
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token for Llama tokenizer")
    
    print(f"Loading model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model.eval()
    
    return model, tokenizer

def main():
    system_prompt = ""

    model_name = os.environ.get("LLM_MODEL_PATH", "")
    if not model_name:
        raise ValueError(
            "LLM model path is not set. "
            "Please export LLM_MODEL_PATH=/path/to/your/llm before running this script."
        )

    input_file = "item_str_prompt.jsonline"
    output_gen_avg_file = "item_gen_avg_emb.jsonline"
    output_input_avg_file = "item_input_avg_emb.jsonline"
    # Item noun used in the prompt suffix, e.g. "fashion item", "beauty product", "business".
    # Override via env var LLM_ITEM_TYPE when switching datasets.
    item_type = os.environ.get("LLM_ITEM_TYPE", "item")
    batch_size = 16

    model, tokenizer = load_model_and_tokenizer(model_name)
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        all_lines = [line.strip() for line in f_in]
    
    with open(output_gen_avg_file, 'w', encoding='utf-8') as f_out_gen_avg, \
         open(output_input_avg_file, 'w', encoding='utf-8') as f_out_input_avg:
        
        for batch_start in range(0, len(all_lines), batch_size):
            batch_end = min(batch_start + batch_size, len(all_lines))
            batch_lines = all_lines[batch_start:batch_end]
            batch_data = []
            batch_prompts = []
            
            for line in batch_lines:
                try:
                    data = json.loads(line)
                    input_content = data.get("input", "")
                    user_prompt = input_content
                    
                    batch_data.append(data)
                    batch_prompts.append((system_prompt, user_prompt))
                except Exception as e:
                    print(f"Error preparing data: {e}")
            
            if batch_prompts:
                batch_results = batch_get_model_responses(model, tokenizer, batch_prompts, item_type)
                
                for i, (data, result) in enumerate(zip(batch_data, batch_results)):
                    if result:
                        gen_avg_data = data.copy()
                        input_avg_data = data.copy()
                        
                        gen_avg_data["response"] = result["full_response"]
                        input_avg_data["response"] = result["full_response"]
                        gen_avg_data["model_input"] = result["model_input"]
                        input_avg_data["model_input"] = result["model_input"]  
                        gen_avg_data["hidden_states"] = result["gen_avg_embedding"]           
                        input_avg_data["hidden_states"] = result["input_avg_embedding"]
                        
                        f_out_gen_avg.write(json.dumps(gen_avg_data, ensure_ascii=False) + '\n')
                        f_out_input_avg.write(json.dumps(input_avg_data, ensure_ascii=False) + '\n')
                        
                        print(f"Processed item {batch_start + i + 1}")
                    else:
                        error_data = {"error": "Processing failed", "item_id": data.get("item_id")}
                        f_out_gen_avg.write(json.dumps(error_data, ensure_ascii=False) + '\n')
                        f_out_input_avg.write(json.dumps(error_data, ensure_ascii=False) + '\n')
                
                f_out_gen_avg.flush()
                f_out_input_avg.flush()

def batch_get_model_responses(model, tokenizer, prompts_list, item_type="item"):
    batch_inputs = []
    original_prompts = []
    
    is_chat_model = "chat" in model.config._name_or_path.lower() or "instruct" in model.config._name_or_path.lower()
    
    for system_prompt, user_prompt in prompts_list:
        user_prompt = user_prompt + f"\nWhich potential users align best with this {item_type}?"
        
        if is_chat_model:
            prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
        else:
            prompt = f"{user_prompt}"
        
        batch_inputs.append(prompt)
        original_prompts.append(prompt) 
    
    model_inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True).to(model.device)
    
    with torch.no_grad():
        torch.cuda.empty_cache()
        
        gen_outputs = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2,
            return_dict_in_generate=True,
            output_hidden_states=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        results = []
        generated_ids = gen_outputs.sequences
        
        for i, output_ids in enumerate(generated_ids):
            input_length = model_inputs.attention_mask[i].sum().item()
            
            generated_ids_only = output_ids[input_length:]
            
            full_output = tokenizer.decode(output_ids, skip_special_tokens=True)
            
            if is_chat_model and "[/INST]" in full_output:
                full_response = full_output.split("[/INST]")[-1].strip()
            else:
                original_prompt_text = original_prompts[i]
                if full_output.startswith(original_prompt_text):
                    full_response = full_output[len(original_prompt_text):].strip()
                else:
                    full_response = tokenizer.decode(generated_ids_only, skip_special_tokens=True).strip()
            
            print(f"Generated response (first 100 chars): {full_response[:100]}...")
            
            attention_mask = model_inputs.attention_mask[i].cpu()
            first_hidden = gen_outputs.hidden_states[0][-1][i].detach().cpu()
            valid_embeddings = first_hidden[attention_mask.bool()]
            input_avg_embedding = torch.mean(valid_embeddings, dim=0, keepdim=True)
            all_gen_hidden_states = []
            
            for step_idx in range(len(generated_ids_only)):
                if step_idx < len(gen_outputs.hidden_states) - 1: 
                    step_hidden = gen_outputs.hidden_states[step_idx + 1][-1][i].detach().cpu()
                    all_gen_hidden_states.append(step_hidden)
            

            gen_avg_embedding = None
            if all_gen_hidden_states:
                gen_stacked_embedding = torch.stack(all_gen_hidden_states, dim=0)
                gen_avg_embedding = torch.mean(gen_stacked_embedding, dim=0, keepdim=True)
            
            results.append({
                "full_response": full_response,
                "model_input": original_prompts[i],
                "input_avg_embedding": input_avg_embedding.tolist() if input_avg_embedding is not None else None,
                "gen_avg_embedding": gen_avg_embedding.tolist() if gen_avg_embedding is not None else None
            })
            
            if i < len(generated_ids) - 1:
                torch.cuda.empty_cache()
                
        torch.cuda.empty_cache()
        return results
if __name__ == "__main__":
    main()
