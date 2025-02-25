import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
embedding_data = torch.load('data.pt')
df = pd.read_json('data2.json')

import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# 모델 및 데이터 로드
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
embedding_data = torch.load('data.pt')
df = pd.read_json('data2.json')


def find_best_matches(question, top_n=3):
    """ 입력된 질문과 가장 유사한 질문을 찾아 top_n 개의 답변을 반환 """
    question = question.replace(" ", "")
    sentence_encode = model.encode(question)
    sentence_tensor = torch.tensor(sentence_encode)

    cos_sim = util.cos_sim(sentence_tensor, embedding_data).squeeze(0)
    top_indices = np.argsort(cos_sim.numpy())[::-1][:top_n]

    results = []
    for idx in top_indices:
        selected_qes = df['Title'][idx]
        selected_qes_encode = model.encode(selected_qes)
        score = np.dot(sentence_tensor, selected_qes_encode) / (
                    np.linalg.norm(sentence_tensor) * np.linalg.norm(selected_qes_encode))
        answer = df['Answer'][idx]
        results.append((selected_qes, score, answer))

    return results


# CLI 챗봇 실행
print("챗봇을 시작합니다. 질문을 입력하세요 (종료하려면 'exit' 입력)")
while True:
    user_input = input("질문: ")
    if user_input.lower() == 'exit':
        print("챗봇을 종료합니다.")
        break

    results = find_best_matches(user_input)
    for i, (selected_qes, score, answer) in enumerate(results, 1):
        print(f"\n[{i}] 유사한 질문: {selected_qes}")
        print(f"유사도 점수: {score:.4f}")
        print(f"답변: {answer}")
    print()


