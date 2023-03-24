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
    """
    This function generates a response to a given prompt using the DialoGPT model.
    The function first encodes the prompt using the tokenizer, adding an end-of-sequence token (eos_token) to the end of the input.
    It then passes the encoded input to the generate() method of the DialoGPT model, which generates one or more responses to the input
    based on the model's learned probabilities. The generate() method takes several parameters, including max_length, temperature, top_k,
    and num_return_sequences, which control various aspects of the response generation process. The function then decodes the generated
    responses using the tokenizer and returns them as a list.

    Arguments: 
    - prompt: The prompt for which a response is being generated.
    - max_length: The maximum length (in tokens) of the generated response.
    - temperature: A value that controls the "creativity" of the generated response. Lower temperatures result in more conservative responses,
    while higher temperatures result in more creative (and potentially nonsensical) responses.
    - top_k: The number of top-scoring tokens to consider when generating the next token. Higher values result in more conservative responses,
    while lower values result in more creative (and potentially nonsensical) responses.
    - num_return_sequences: The number of alternative responses to generate.
    """
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
