import requests

API_KEY = "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"
BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

def call_llm(messages):
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": MODEL,
            "messages": messages,
            "temperature": 0.2,  # Low temperature for consistency
            "max_tokens": 256
        }
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]

def get_keywords(user_input: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Python expert assistant. Your job is to infer and return all Python-related keywords, libraries, modules, functions, and classes "
                "that would typically appear in code fulfilling the user's request.\n\n"
                "Important:\n"
                "- Return ONLY a comma-separated list of keywords\n"
                "- Do NOT include explanations, bullet points, or code blocks\n"
                "- Keep all relevant keywords, especially those from standard or popular libraries like Pygame\n"
            )
        },
        {
            "role": "user",
            "content": user_input
        }
    ]
    return call_llm(messages)




