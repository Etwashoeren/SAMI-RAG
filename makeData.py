import json

from openaicall import LlmClient

llmClient = LlmClient()

def main() -> None:

    file_path = "Haksa.json"

    # 파일 열기
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

        # 글 N개 순회
        for i, row in enumerate(data):

            title = row["title"]
            start = row["start"]
            end = row["end"]

            # LLM한테 시킬 프롬포트 작성
            system_prompt = f"""
                조건 : 
                 1. 나는 {file_path}를 기반으로 데이터셋을 만들고 있어.
                 2. {file_path}는 상명대학교의 학사일정이야.
                 3. {file_path}는 title(일정 이름)과 start(시작일), 그리고 end(종료일)로 이루어져 있어.

                예시로 : 
                Title : {file_path}의 title
                Answer : {start}부터 {end}까지 입니다.
                Title : {file_path}의 title
                Answer : {start}부터 {end}까지 입니다.

                아래 부터는 내용 데이터야 : 
            """

            # 글 : 제목, 내용
            question_prompt = f"""
                title : {title}
                start : {start}
                end : {end}
            """

            # LLM 요청, 응답
            response = llmClient.call_llm(system_prompt, question_prompt)
            answer = response.choices[0].message.content

            # 결과 txt 파일에 추가
            with open("output2.txt", "a", encoding="utf-8") as f:
                questions_answers = answer.strip().split("\n")
                for item in questions_answers:
                    if item.strip():
                        f.write(item.strip() + "\n")

            print(i, title, "완료")

if __name__ == "__main__":
    main()