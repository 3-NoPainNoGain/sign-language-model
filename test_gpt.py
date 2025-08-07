# test_gpt.py
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

res = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "안녕하세요를 자연스럽게 말해줘"}
    ]
)

print(res.choices[0].message.content)
