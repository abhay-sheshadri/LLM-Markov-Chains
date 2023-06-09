from transformers import GPTNeoXForCausalLM, AutoTokenizer
import transformers
import torch


def load_model(name):
    # Load model from HuggingFace Hub
    model = GPTNeoXForCausalLM.from_pretrained(
        name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        device_map="auto",
    )
    return model, tokenizer


def generate_tokens(
    model: GPTNeoXForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    n_tokens: int,
    context_window: int = 2048,
    top_k: int = 10
):
    # Tokenize inputs
    inputs = tokenizer(prompt, return_tensors='pt')
    full_tokens = [inputs["input_ids"]]
    # Sample n_tokens
    for i in range(n_tokens):
        outputs = model(**inputs)
        tokens = outputs.logits[:, -1:].argmax(-1)
        full_tokens.append(tokens.cpu())
        inputs["input_ids"] = torch.cat( (inputs["input_ids"], tokens), dim=-1)
        inputs["attention_mask"] = torch.cat( (inputs["attention_mask"], torch.ones_like(tokens) ), dim=-1)
        inputs["input_ids"] = inputs["input_ids"][:, -context_window:]
        inputs["attention_mask"] = inputs["attention_mask"][:, -context_window:]
        #  inputs["past_key_values"] = outputs["past_key_values"]
    full_tokens = torch.cat(full_tokens, dim=-1)
    return full_tokens
    #return tokenizer.decode(full_tokens[0], skip_special_tokens=True)