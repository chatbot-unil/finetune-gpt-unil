import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
import json
from datetime import datetime

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
system_message = os.getenv("SYSTEM_MESSAGE")

client = OpenAI()

def get_last_fine_tuned_model():
    list_models = client.fine_tuning.jobs.list(limit=5)
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
    dict_qa = {}
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            json_line = json.loads(line)
            if 'messages' in json_line and len(json_line['messages']) > 1:
               dict_qa[json_line['messages'][1]['content']] = json_line['messages'][2]['content']
    return dict_qa


if __name__ == '__main__':
    dict_qa = load_jsonl_file('data/training_data.jsonl')
    
    if len(sys.argv) > 1:
        model_id = sys.argv[1]
    else:
        model_id = get_last_fine_tuned_model()

    counter = 0
    for question, expected_response in dict_qa.items():
        response = completions(question, model_id)
        print(f"Question: {question}")
        print(f"Response: {response}")
        print(f"Response attendue: {expected_response}")
        counter += 1
        if counter >= 10:
            break