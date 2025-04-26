import openai
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def get_conversation_topic(conversation_text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Summarize the MAIN TOPIC of a user conversation in 3-5 words."},
                {"role": "user", "content": f"Conversation:\n{conversation_text}\n\nTopic?"}
            ],
            temperature=0.2,
            max_tokens=30
        )
        topic = response['choices'][0]['message']['content'].strip()
        return topic
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "Unknown topic"
