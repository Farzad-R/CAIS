"""
This is a fun project using DialoGPT-large to design a chatbot.
Attention: conversations won't make sense. This code is just for demonstration purposes.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import random

# Load the model and the tokenizer 
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")


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

def generate_response(prompt, max_length=100, temperature=0.7, top_k=50, repetition_penalty=1.2, num_return_sequences=3):
    # Encode the user input using the tokenizer
    input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")
    # new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')
    # append the new user input tokens to the chat history
    # bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
    # chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    # Generate a response using the model
    output = model.generate(
        input_ids=input_ids,
        max_length=max_length + len(input_ids[0]),
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
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
        responses = generate_response(user_input)
        response = random.choice(responses)
    except:
        response = "I'm sorry, I didn't understand what you said. Can you please rephrase?"
    
    print(" Chatbot: ", response)