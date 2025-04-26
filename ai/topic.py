import openai

openai.api_key = "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"

def get_conversation_topic(conversation_text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Summarize the MAIN TOPIC of a user conversation in 3-5 words."},
                {"role": "user", "content": f"Conversation:\n{conversation_text}\n\nTopic?"}
            ],
            temperature=0.2,  # Makes it more consistent
            max_tokens=30
        )
        topic = response['choices'][0]['message']['content'].strip()
        return topic
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "Unknown topic"
