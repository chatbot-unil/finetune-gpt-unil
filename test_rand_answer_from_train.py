import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
import json
import random
from datetime import datetime

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
system_message = os.getenv("SYSTEM_MESSAGE")

client = OpenAI()

def get_last_fine_tuned_model():
	list_models = client.fine_tuning.jobs.list(limit=2)
	for model in list_models.data:
		if model.status == 'succeeded':
			created_at = datetime.utcfromtimestamp(model.created_at).strftime('%Y-%m-%d %H:%M:%S')
			print(f"Model name: {model.fine_tuned_model} - Created at: {created_at}")
			return model.fine_tuned_model
	return None

def completions(message, model_id):
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
    model_id = get_last_fine_tuned_model()
    if len(training_data) >= 10:
        random_questions = random.sample(training_data, 10)
        for question in random_questions:
            response = completions(question, model_id)
            print(f"Question: {question}")
            print(f"Response: {response}")
    else:
        print("Not enough data to select 10 random questions.")