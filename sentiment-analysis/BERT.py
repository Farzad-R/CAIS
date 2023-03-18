# Dataset: https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset
#%%
import transformers
import torch
import pandas as pd
from tqdm import tqdm
from pyprojroot import here
from torch.utils.data import TensorDataset, DataLoader
epochs = 1
learning_rate = 5e-5
batch_size = 16

# Check if GPU is available or not and set the device
if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
#%%
# Load data for sentiment analysis
data = pd.read_csv(here('data/sentient_analysis/train.csv'), encoding_errors='replace')
data = data.dropna()
print(data.shape)

# Get number of unique labels
num_labels = len(data["sentiment"].unique())
print("num_labels:", num_labels)

sentences = data['text'].values.tolist()
print("Length sentences:", len(sentences))

# Onehotencode the labels
labels = pd.get_dummies(data['sentiment']).values
print("Length labels:", len(labels))
#%%
# Load pre-trained BERT model
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
# move the model to device
model.to(device)
#%%
# Tokenize and encode the sentences
encoded_inputs = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')

# Convert data to PyTorch TensorDataset
dataset = TensorDataset(encoded_inputs['input_ids'], encoded_inputs['attention_mask'], torch.from_numpy(labels).float())

# Create DataLoader to generate batches
dataloader = DataLoader(dataset, batch_size=batch_size)

del (data, labels, sentences)
#%%
# Fine-tune the model on the sentiment analysis task
optimizer = transformers.AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.BCEWithLogitsLoss()

#%%
model.train()
for epoch in range(epochs):
    epoch_loss = 0
    for batch in tqdm(dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels_tensor = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels_tensor)
        loss = loss_fn(outputs.logits, labels_tensor)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch} Loss: {epoch_loss/len(dataloader)}")
#%%
# Test the model on new data
new_sentences = ["This movie was great!", "This product is terrible!", "I am neutral about this."]
encoded_inputs = tokenizer(new_sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
encoded_inputs.to(device)

classes = {0:"Negative", 1:"Neutral", 2:"Positive"}

model.eval()
with torch.no_grad():
    outputs = model(**encoded_inputs)
    predictions = torch.softmax(outputs.logits, dim=1).argmax(dim=1)

print(predictions)
for i in predictions:
    # print(i.item())
    print("Results:", classes[i.item()]) 
