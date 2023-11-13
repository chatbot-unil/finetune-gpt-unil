import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
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

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("Usage: python script.py <question>")
		sys.exit(1)

	question = sys.argv[1]
	model_id = get_last_fine_tuned_model()
	response = completions(question, model_id)
	print(response)