from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

"""
we'll use GPT-2 to generate text based on an open-source dataset of movie summaries:

- IMDb dataset: contains movie summaries and other information about movies. 
- We're only using the first 10% of the dataset for demonstration purposes.
"""
# Load the movie summary dataset
dataset = load_dataset("imdb", split="train[:10%]")

# Load the pre-trained GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")


def generate_text(prompt, max_length=100):
    # Encode the prompt using the tokenizer
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate text using the model
    output = model.generate(input_ids=input_ids, max_length=max_length, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the generated text using the tokenizer
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return output_text


# Generate a sample movie summary
summary = generate_text("If you want to teach me how to stand on my feet \n")
print(summary)