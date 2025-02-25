from openai import OpenAI

class LlmClient():
    def __init__(self):

        self.client = OpenAI(
            api_key="[API-KEY]",
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
