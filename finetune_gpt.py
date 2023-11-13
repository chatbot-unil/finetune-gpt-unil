import os
import json
import random
import sys
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load and merge all the JSON files
json_files = os.listdir('data/json')
training_data = []
validating_data = []
for json_file in json_files:
	with open(f'data/json/{json_file}', 'r', encoding='utf-8') as f:
		data = json.load(f)
		training_data.extend(data)

# validation_data is 20% of the training data
validating_data = training_data[:int(len(training_data) * 0.2)]

random.shuffle(training_data)
random.shuffle(validating_data)

# Save the data to a jsonl file
with open('data/training_data.jsonl', 'w', encoding='utf-8') as f:
	for example in training_data:
		json.dump(example, f, ensure_ascii=False)
		f.write('\n')

with open('data/validating_data.jsonl', 'w', encoding='utf-8') as f:
	for example in validating_data:
		json.dump(example, f, ensure_ascii=False)
		f.write('\n')

print(f"Training on {len(training_data)} examples")
print(f"Validating on {len(validating_data)} examples")

# Load the model
client = OpenAI()

file_training = client.files.create(
  file=open("data/training_data.jsonl", "rb"),
  purpose="fine-tune",
)

file_validating = client.files.create(
	file=open("data/validating_data.jsonl", "rb"),
  	purpose="fine-tune",
)

# Start the fine-tuning and get the ID of the fine-tuning job
print("Starting fine-tuning...")
fine_tune = client.fine_tuning.jobs.create(
	training_file=file_training.id,
	validation_file=file_validating.id,
	model="gpt-3.5-turbo-1106",
)

# Wait for the fine-tuning to complete
print("Waiting for fine-tuning to complete...")
while True:
	time.sleep(5)
	fine_tune = client.fine_tuning.jobs.retrieve(fine_tune.id)
	if fine_tune.status == "succeeded":
		print("Fine-tuning completed successfully!")
		break
	if fine_tune.status == "failed":
		print("Fine-tuning failed")
		sys.exit(1)
print("Fine-tuning completed!")

# Save the model ID to the .env file is the MODEL_ID variable is not already set or it modified 
with open('.env', 'a') as f:
	if os.getenv("MODEL_ID") is None or os.getenv("MODEL_ID") != fine_tune.fine_tuned_model:
		f.write(f"MODEL_ID={fine_tune.fine_tuned_model}\n")
		print("Model ID saved to .env file")
	else:
		print("Model ID already saved to .env file")