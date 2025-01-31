import pandas as pd

# Define the file path of the Excel file
# file_path = "../../code/llm/LLaVA-13b/code/imagenet2ktrain.csv"
file_path = "../../code/llm/LLaVA-13b/code/sexyprompt.csv"

# Load the Excel file
df = pd.read_csv(file_path)

# Extract the 'prompt' column
prompts = df['prompt']

# Convert the prompts to a single text with each prompt on a new line
prompts_text = "\n".join(prompts)

# Define the output file path
# output_path = "../../code/imagenet-autoencoder-main/list/2kimagenettrainprompt.txt"
output_path = "../../code/imagenet-autoencoder-main/list/sexytrainprompt.txt"

# Save the text to a file
with open(output_path, "w") as file:
    file.write(prompts_text)

print(f"Prompts saved to {output_path}")
