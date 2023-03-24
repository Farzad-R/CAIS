"""
This is a fun project using DialoGPT-medium to design a chatbot.
Attention: conversations won't make sense. This code is just for demonstration purposes.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and the tokenizer 
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")


def filter_response(prompt, response):
    # Split the response into individual sentences
    sentences = response.split(".")
    
    # Use a machine learning model to rank the sentences based on their relevance to the prompt
    # This requires training a model on a dataset of conversational data
    # Here, we'll just use a simple heuristic of picking the longest sentence that contains the prompt
    filtered_sentences = [s.strip() for s in sentences if prompt.lower() in s.lower()]
    filtered_sentences = sorted(filtered_sentences, key=len, reverse=True)
    
    # If there are no relevant sentences, return the entire response
    if len(filtered_sentences) == 0:
        return response
    
    # Otherwise, return the most relevant sentence
    return filtered_sentences[0]

def generate_response(prompt, max_length=100, temperature=0.7, top_k=50, num_return_sequences=3):
    # Encode the user input using the tokenizer
    input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")
    
    # Generate a response using the model
    output = model.generate(
        input_ids=input_ids,
        max_length=max_length + len(input_ids[0]),
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=1.5,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=num_return_sequences
    )
    
    # Decode the generated response using the tokenizer
    responses = [tokenizer.decode(o, skip_special_tokens=True) for o in output]
    
    # Filter the responses based on the user input
    filtered_responses = [filter_response(prompt, r) for r in responses]
    
    return filtered_responses


prompt = "Hello, how can I help you today?"

# Start the conversation loop
while True:
    # Get user input
    user_input = input("You: ")
    
    # Check if the user wants to end the conversation
    if user_input.lower() in ["bye", "goodbye"]:
        print("Chatbot: Goodbye!")
        break
    
    # Generate a response
    try:
        response = generate_response(user_input)[0]
    except:
        response = "I'm sorry, I didn't understand what you said. Can you please rephrase?"
    
    
    print(" Chatbot: ", response)
