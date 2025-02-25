from openai import OpenAI

class LlmClient():
    def __init__(self):

        self.client = OpenAI(
            api_key="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",
        )
        self.model = "gpt-4o-mini"

    def call_llm(self, system_prompt:str, user_prompt:str) -> str:

        return self.client.chat.completions.create(
            model=self.model,
            messages= [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            seed= 42
        )

llmClient = LlmClient()

system = f""
question = f""

response = llmClient.call_llm(system, question)
print(response.choices[0].message.content)