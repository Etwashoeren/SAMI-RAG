import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
embedding_data = torch.load('data.pt')
df = pd.read_json('data2.json')

def find_best_match(question):
    """ 입력된 질문과 가장 유사한 질문을 찾아 답변을 반환 """
    question = question.replace(" ", "")
    sentence_encode = model.encode(question)
    sentence_tensor = torch.tensor(sentence_encode)

    cos_sim = util.cos_sim(sentence_tensor, embedding_data)
    best_sim_idx = int(np.argmax(cos_sim))

    selected_qes = df['Title'][best_sim_idx]
    selected_qes_encode = model.encode(selected_qes)
    score = np.dot(sentence_tensor, selected_qes_encode) / (
                np.linalg.norm(sentence_tensor) * np.linalg.norm(selected_qes_encode))

    answer = df['Answer'][best_sim_idx]
    return selected_qes, score, answer


# CLI 챗봇 실행
print("챗봇을 시작합니다. 질문을 입력하세요 (종료하려면 'exit' 입력)")
while True:
    user_input = input("질문: ")
    if user_input.lower() == 'exit':
        print("챗봇을 종료합니다.")
        break

    selected_qes, score, answer = find_best_match(user_input)
    print(f"\n가장 유사한 질문: {selected_qes}")
    print(f"유사도 점수: {score:.4f}")
    print(f"답변: {answer}\n")
