import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

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

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("Usage: python script.py <question>")
		sys.exit(1)

	question = sys.argv[1]
	response = completions(question)
	print(response)