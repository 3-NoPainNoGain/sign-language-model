import os
from dotenv import load_dotenv
load_dotenv()

MODEL = "gpt-3.5-turbo"

def complete_sentence(prompt: str) -> str:
    sys = "너는 병원에서 청각장애인 환자가 입력한 단어를 조사를 이용해서 자연스럽게 정렬하는 한국어 생성기다. 무조건 한국어 자연스러운 문장(명사->동사)으로 만들되 새 단어 추가 금지. 최대 2단어까지 연결"
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content":sys},{"role":"user","content":prompt}],
            temperature=0.2,
            max_tokens=60,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        resp = openai.ChatCompletion.create(
            model=MODEL,
            messages=[{"role":"system","content":sys},{"role":"user","content":prompt}],
            temperature=0.2,
            max_tokens=60,
        )
        return resp.choices[0].message.content.strip()
