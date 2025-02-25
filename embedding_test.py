import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
embedding_data = torch.load('data.pt')
df = pd.read_json('data2.json')

sentence = "수강 신청이 언제야?"
print("질문 문장: ", sentence)
sentence = sentence.replace(" ", "")
print("공백 제거 문장: ", sentence)

sentence_encode = model.encode(sentence)
sentence_tensor = torch.tensor(sentence_encode)

cos_sim = util.cos_sim(sentence_tensor, embedding_data)
print(f"가장 높은 코사인 유사도 idx: {int(np.argmax(cos_sim))}")

best_sim_idx = int(np.argmax(cos_sim))
selected_qes = df['Title'][best_sim_idx]
print(f"선택된 질문 = {selected_qes}")

selected_qes_encode = model.encode(selected_qes)
score = np.dot(sentence_tensor, selected_qes_encode) / (np.linalg.norm(sentence_tensor) * np.linalg.norm(selected_qes_encode))
print(f"선탠된 질문과의 유사도 = {score}")

answer = df['Answer'][best_sim_idx]
print(f"\n답변 : {answer}\n")