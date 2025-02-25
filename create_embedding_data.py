import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import torch
from sentence_transformers import SentenceTransformer

train_file = "data.json"
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

df = pd.read_json(train_file)
df['embedding_vector'] = df['Title'].progress_map(lambda x: model.encode(x))
df.to_json('data2.json', index=False)

embedding_data = torch.tensor(df['embedding_vector'].tolist())
torch.save(embedding_data, 'data.pt')