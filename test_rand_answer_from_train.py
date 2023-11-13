import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
import json
import random

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
model_id = os.getenv("MODEL_ID")
system_message = os.getenv("SYSTEM_MESSAGE")

client = OpenAI()

def completions(message):
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": message},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content

def load_jsonl_file(path):
    answer_list = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            # Parse the JSON object from the line
            json_line = json.loads(line)
            # Now you can access the 'messages' key
            if 'messages' in json_line and len(json_line['messages']) > 1:
                answer_list.append(json_line['messages'][1]['content'])
    return answer_list


if __name__ == '__main__':
    training_data = load_jsonl_file('data/training_data.jsonl')
    if len(training_data) >= 10:
        random_questions = random.sample(training_data, 10)
        for question in random_questions:
            response = completions(question)
            print(f"Question: {question}")
            print(f"Response: {response}")
    else:
        print("Not enough data to select 10 random questions.")