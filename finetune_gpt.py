import os
import argparse
from dotenv import load_dotenv
from openai import OpenAI


parser = argparse.ArgumentParser(description="Fine-tune OpenAI model.")
parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
parser.add_argument('--training_data_path', type=str, default='data/training_data.jsonl', help='Path to training data')
parser.add_argument('--validating_data_path', type=str, default='data/validating_data.jsonl', help='Path to validating data')
args = parser.parse_args()

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load the model
client = OpenAI()

file_training = client.files.create(
  file=open(args.training_data_path, "rb"),
  purpose="fine-tune",
)

file_validating = client.files.create(
	file=open(args.validating_data_path, "rb"),
  	purpose="fine-tune",
)

# Start the fine-tuning and get the ID of the fine-tuning job
print("Starting fine-tuning...")
fine_tune = client.fine_tuning.jobs.create(
	training_file=file_training.id,
	validation_file=file_validating.id,
	model="gpt-3.5-turbo-1106",
	hyperparameters={
      "n_epochs": args.epochs,
    }
)