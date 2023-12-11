import os
import re
import json
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(description="Explore logs process.")
parser.add_argument('--logs', type=str, default='logs/', help='Log file name')
parser.add_argument('--save_dir', type=str, default='plots/', help='Save directory')

args = parser.parse_args()

def get_all_logs(path):
    logs = []
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            if re.search("results", file):
                logs.append(os.path.join(path, file))
        elif os.path.isdir(os.path.join(path, file)):
            if re.search("old", file):
                continue
            logs.extend(get_all_logs(os.path.join(path, file)))
    return logs

def get_log_file(log_name):
    log_file = open(os.path.join(log_name), "r")
    return log_file

def precision_moyenne_from_json(log_file):
    data = json.load(log_file)
    for item in data:
        if 'precision_moyenne' in item and 'model_id' in item:
            return item['precision_moyenne'], item['model_id']
    return None, None

def isolate_precisions_for_model(log_file, model_id_to_match):
    data = json.load(log_file)
    precisions = []
    for item in data:
        if 'files' in item and item.get('model_id') == model_id_to_match:
            for file_entry in item['files']:
                if 'precisions' in file_entry:
                    precisions.append(file_entry['precisions'])
    return precisions

def plot_boxplots(daily_model_precisions, save_dir):
    for date, model_data in daily_model_precisions.items():
        num_subplots = len(model_data)
        if num_subplots == 0:
            continue  # Skip if there are no precisions for this date
        
        num_cols = 3  # Number of columns for subplots
        num_rows = (num_subplots + num_cols - 1) // num_cols  # Calculate the number of rows needed
        
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))  # Adjust the figsize as needed
        fig.suptitle(f'Boxplot of Model Precisions\nDate: {date}')

        # Flatten the axs array if there's only one row
        if num_rows == 1:
            axs = axs.reshape(1, -1)

        for i, (model_id, model_info) in enumerate(model_data.items()):
            row_idx = i // num_cols
            col_idx = i % num_cols
            ax = axs[row_idx, col_idx]

            if not model_info['precisions']:
                continue  # Skip empty precision lists

            ax.boxplot(model_info['precisions'], labels=[model_id])
            ax.set_title(f'Model ID: {model_id}\nPurpose: {model_info["purpose"]}', fontsize=10)
            ax.set_ylabel('Precision')
            ax.set_ylim(0, 100)  # Set y-axis limits to 0-100

            # Calculate and annotate the average precision
            average_precision = sum(model_info['precisions']) / len(model_info['precisions'])
            ax.set_xlabel(f'Avg: {average_precision:.2f}')

        # Remove empty subplots if there are fewer than 3 subplots in the last row
        for i in range(len(model_data), num_rows * num_cols):
            row_idx = i // num_cols
            col_idx = i % num_cols
            fig.delaxes(axs[row_idx, col_idx])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(save_dir, f"{date}.png")
        plt.savefig(save_path)
        plt.close(fig)

if __name__ == "__main__":
    logs = get_all_logs(args.logs)
    daily_model_precisions = {}

    print(f"Found {len(logs)} logs")

    for log in logs:
        print(f"Processing {log}")
        with get_log_file(log) as log_file:
            data = json.load(log_file)
            print(f"Found {len(data)} items in {log}")
            print(f"Last item: {data[-1]}")

            # Extract global date and purpose
            purpose = data[-1].get('purpose', "Default Purpose")
            date_str = data[-1].get('date', "Unknown Date").split(' ')[0]
            date = datetime.strptime(date_str, "%d/%m/%Y").strftime("%Y-%m-%d")

            for item in data[:-1]:  # Skip the last item (purpose and date)
                model_id = item['model_id']
                if date not in daily_model_precisions:
                    daily_model_precisions[date] = {}
                if model_id not in daily_model_precisions[date]:
                    daily_model_precisions[date][model_id] = {'precisions': [], 'purpose': purpose}

                # Check if 'files' key exists in the item
                if 'files' in item:
                    for file_entry in item['files']:
                        if 'precisions' in file_entry:
                            daily_model_precisions[date][model_id]['precisions'].append(file_entry['precisions'])
                else:
                    print(f"'files' key not found in item: {item}")
                    
    plot_boxplots(daily_model_precisions, args.save_dir)
