from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("API key is missing! Please set it in the .env file.")

client = Groq(api_key=api_key)
completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
           "content": "Hello! You are a personalized health assistant and Kunal's AI."

        },
        {
            "role": "user",
            "content": "hii"
        },
        {
            "role": "assistant",
            "content": "Hello Kunal! How are you doing today? Is there anything I can help you with regarding your health or wellness?"
        },
        {
            "role": "user",
            "content": "yes\n"
        },
        {
            "role": "assistant",
            "content": "I'm here to help, Kunal. What's on your mind? Do you have a specific health concern, or would you like to track your progress, set goals, or discuss a particular topic, such as nutrition, fitness, or stress management? Let me know, and I'll do my best to assist you."
        },
        {
            "role": "user",
            "content": "nothing"
        },
        {
            "role": "assistant",
            "content": "Sometimes it's nice"
        }
    ],
    temperature=1,
    max_completion_tokens=1000,
    top_p=1,
    stream=True,
    stop=None,
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
