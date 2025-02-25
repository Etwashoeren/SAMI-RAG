import json

def main(txt_file_path: str, json_file_path: str) -> None:

    qa_list = []

    with open(txt_file_path, "r", encoding="utf-8") as txt_file:
        lines = txt_file.readlines()

    current_question = None
    current_answer = []

    for line in lines:
        line = line.strip()
        if line.startswith("Question :"):
            if current_question and current_answer:
                qa_list.append({
                    "Question": current_question,
                    "Answer": "\n".join(current_answer).strip()
                })
                current_question = None
                current_answer = []
            current_question = line.replace("Question :", "").strip()
        elif line.startswith("Answer :"):
            current_answer.append(line.replace("Answer :", "").strip())
        elif current_question and current_answer:
            current_answer.append(line)

    if current_question and current_answer:
        qa_list.append({
            "Question": current_question,
            "Answer": "\n".join(current_answer).strip()
        })

    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(qa_list, json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main("output.txt", "output.json")