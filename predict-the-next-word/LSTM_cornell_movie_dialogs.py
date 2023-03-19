# Dataset: https://www.kaggle.com/datasets/Cornell-University/movie-dialog-corpus
# data load and preprocessing from: https://www.kaggle.com/code/malyvsen/chatbot-movie-dialogs

from collections import Counter
import random
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm, trange
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from pyprojroot import here

# Load the data movie_line_data
line_data = pd.read_csv(here('data//movie_lines/movie_lines.tsv'), encoding='utf-8-sig',header=None)
line_data = line_data[0].str.split('\t').to_list()

line_data = [l for l in line_data if len(l) == 5]
print(f'in total {len(line_data)} utterances')
line_data[:4]

lines = pd.DataFrame(line_data, columns=['line_id', 'speaker_id', 'movie_id', 'speaker_name', 'text'])
lines = lines.set_index('line_id')
lines.loc['L1000']

conversation_data = pd.read_csv(here('data/archive/movie_conversations.tsv'), encoding='utf-8-sig', sep='\t', header=None)
conversation_data = conversation_data.rename(columns={0: 'speaker1_id', 1: 'speaker2_id', 2: 'movie_id', 3: 'line_ids'})
conversation_data

def build_conversation(line_ids):
    id_list = line_ids[1:-2].replace('\'', '').split(' ')
    def build_utterance(line):
        return (line.speaker_id, line.text)
    try:
        return [build_utterance(lines.loc[line_id]) for line_id in id_list]
    except KeyError:
        return []

conversations = [build_conversation(line_ids) for line_ids in tqdm(conversation_data.line_ids)]
print(f'in total {len(conversations)} conversations')
conversations[:4]

# Filtering the data
plt.title('distribution of conversation lengths')
plt.xlabel('number of utterances')
plt.ylabel('number of conversations')
plt.hist([len(c) for c in conversations], bins=20, range=(0, 20))
plt.show()

conversations = [c for c in conversations if len(c) > 2]
print(f'{len(conversations)} conversations remain')

plt.title('distribution of conversation character counts')
plt.xlabel('number of characters')
plt.ylabel('number of conversations')
plt.hist([sum(len(u[1]) for u in c) for c in conversations], bins=64, range=(0, 2000))
plt.show()

min_char_count = 128
conversations = [c for c in conversations if sum(len(u[1]) for u in c) > min_char_count]
print(f'{len(conversations)} conversations remain')

def speaker_repeated(conversation):
    for idx in range(1, len(conversation)):
        if conversation[idx][0] == conversation[idx-1][0]:
            return True
    return False

conversations = [conv for conv in conversations if not speaker_repeated(conv)]
print(f'{len(conversations)} conversations remain')

print(f'in total {sum(sum(len(u[1]) for u in c) for c in conversations)} characters in conversations')

## Tokenizing
char_usage = Counter(''.join([utterance[1] for conv in conversations for utterance in conv]))
chars_by_frequency = sorted(char_usage, key=lambda char: -char_usage[char])
[(char, char_usage[char]) for char in chars_by_frequency]

frequent_chars = [char for char in sorted(char_usage) if char_usage[char] >= 100]
print(f'{len(frequent_chars)} frequent chars: {frequent_chars}')


empty_token = len(frequent_chars)
unknown_token = len(frequent_chars) + 1
speaker_change_token = len(frequent_chars) + 2
num_tokens = len(frequent_chars) + 3

def char_to_token(char):
    if char not in frequent_chars:
        return unknown_token
    return frequent_chars.index(char)

def utterance_to_tokens(utterance):
    return [char_to_token(char) for char in utterance[1]] + [speaker_change_token]

def conversation_to_tokens(conversation):
    result = []
    for utterance in conversation:
        result += utterance_to_tokens(utterance)
    return result

conversations_tokenized = [conversation_to_tokens(conversation) for conversation in tqdm(conversations)]
print(f'example conversation, tokenized: {conversations_tokenized[0]}')

def pad_empty(tokens, desired_length):
    return [empty_token] * (desired_length - len(tokens)) + tokens

def random_example(hist_length):
    '''
    Returns tuple of:
        * hist_length tokens
        * the token that followed them directly
    '''
    conversation = random.choice(conversations_tokenized)
    start = random.randint(-hist_length, len(conversation) - hist_length - 1)
    end = start + hist_length
    hist = conversation[max(start, 0):end]
    hist = pad_empty(hist, hist_length)
    return (tf.constant(hist), tf.constant(conversation[end]))

print(random_example(32))


def generate_examples(batch_size, hist_length, labelled=True):
    while True:
        examples = [random_example(hist_length) for _ in range(batch_size)]
        inputs = tf.stack([ex[0] for ex in examples])
        targets = tf.stack([ex[1] for ex in examples])
        if labelled:
            yield inputs, targets
        else:
            yield inputs

iter(generate_examples(2, 16)).__next__()

# Model training
model = tf.keras.models.Sequential[
    tf.keras.layers.Embedding(input_dim=num_tokens, output_dim=64),
    tf.keras.layers.LSTM(units=512, return_sequences=True, input_shape=(None, num_tokens)),
    tf.keras.layers.LSTM(units=256),
    tf.keras.layers.Dense(num_tokens)
    ]
model.summary()

# Compile the model
model.compile(optimizer=Adam(),
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

# Train the model
training_history = model.fit_generator(
    generate_examples(batch_size=128, hist_length=128), 
    epochs=32, 
    steps_per_epoch=1024
    )

plt.title('accuracy vs epoch number')
plt.xlabel('epoch number')
plt.ylabel('accuracy')
plt.plot(training_history.history['sparse_categorical_accuracy'])
plt.show()

## Evaluation
def logits_to_tokens(logits, temperature=1.0):
    one_hot = tf.nn.softmax(logits / temperature)
    result = []
    for row in one_hot:
        result += random.choices(population=list(range(num_tokens)), weights=row.numpy(), k=1)
    return result


def token_to_char(token):
    if token < len(frequent_chars):
        return frequent_chars[token]
    raise ValueError('special token - not a character')


def continue_tokens(tokens, max_length=256, temperature=1.0):
    '''Add a new utterance to the (tokenized) conversation'''
    result = tokens.copy()
    result = pad_empty(result, 16)
    
    for _ in trange(max_length):
        logits = model.predict([[tf.constant(result)]])
        token = logits_to_tokens(logits, temperature=temperature)[0]
        if token == speaker_change_token:
            break
        result.append(token)
    
    result.append(speaker_change_token)
    return result


def tokens_to_conversation(tokens):
    result = []
    to_process = tokens.copy()
    while len(to_process) > 0:
        text = ''
        while len(to_process) > 0:
            token = to_process.pop(0)
            if token == speaker_change_token:
                break
            try:
                text += token_to_char(token)
            except ValueError:
                pass
        result.append((str(len(result) % 2), text))
    return result


def continue_conversation(conversation, max_length=256, temperature=1.0):
    '''Like continue_tokens, but operates on structured conversations'''
    tokens = continue_tokens(conversation_to_tokens(conversation), max_length=max_length, temperature=temperature)
    return tokens_to_conversation(tokens)

for conversation in conversations[:4]:
    print(continue_conversation(conversation, max_length=256))