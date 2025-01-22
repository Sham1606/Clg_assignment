import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import math
import os

# Check for required packages
try:
    import accelerate
except ImportError:
    raise ImportError(
        "Please install the required packages by running: pip install transformers[torch] accelerate>=0.26.0")

# Load pre-trained model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Add padding token to the tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Add special tokens for different sections
special_tokens = ['<image>', '<description>', '<analysis>', '<severity>', '<treatment>']
tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
model.resize_token_embeddings(len(tokenizer))


# Prepare dataset
def load_dataset(file_path, tokenizer):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split the text into individual cases
    cases = text.split('\n\n')

    # Process each case
    processed_cases = []
    for case in cases:
        lines = case.split('\n')
        if len(lines) >= 5:
            processed_case = (
                f"<image>{lines[0]}\n"
                f"<description>{lines[1]}\n"
                f"<analysis>{lines[2]}\n"
                f"<severity>{lines[3]}\n"
                f"<treatment>{' '.join(lines[4:])}"
            )
            processed_cases.append({"text": processed_case})
        else:
            print(f"Warning: Skipping incomplete case: {case}")

    # Convert processed cases into a Dataset object
    dataset = Dataset.from_list(processed_cases)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    return tokenized_dataset


# Check if the dataset file exists
dataset_file = "pneumonia_dataset.txt"
if not os.path.exists(dataset_file):
    print(f"Error: The file {dataset_file} does not exist.")
    print("Please make sure you have created the pneumonia_dataset.txt file in the same directory as this script.")
    exit(1)

# Load the dataset
full_dataset = load_dataset(dataset_file, tokenizer)

# Check the size of the dataset and adjust splitting logic
if len(full_dataset) < 5:
    print(f"Warning: Dataset is too small ({len(full_dataset)} samples) for meaningful splitting.")
    print("Using the entire dataset for both training and evaluation.")
    train_dataset = full_dataset
    eval_dataset = full_dataset
else:
    # Split the dataset
    train_testvalid = full_dataset.train_test_split(test_size=0.2, seed=42)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)

    train_dataset = train_testvalid['train']
    eval_dataset = test_valid['train']
    test_dataset = test_valid['test']

# Set up data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-pneumonia-diagnosis",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    learning_rate=5e-5,
    warmup_steps=100,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Fine-tune the model
print("Starting model fine-tuning...")
trainer.train()
print("Model fine-tuning completed.")

# Evaluate (perplexity)
print("Evaluating the model...")
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


# Generate text
def generate_text(prompt, max_length=300):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, temperature=0.7, top_k=50,
                            top_p=0.95)
    return tokenizer.decode(output[0], skip_special_tokens=False)


# Save the fine-tuned model
print("Saving the fine-tuned model...")
model.save_pretrained("./gpt2-pneumonia-diagnosis")
tokenizer.save_pretrained("./gpt2-pneumonia-diagnosis")
print("Fine-tuned model saved successfully.")


# Function to generate a pneumonia analysis report from an image description
def generate_pneumonia_report(image_description):
    prompt = f"<image>Chest X-ray\n<description>{image_description}\n"
    generated_text = generate_text(prompt, max_length=500)
    return generated_text


# Example usage
if __name__ == "__main__":
    test_description = "The chest X-ray shows patchy opacities in both lower lobes, with air bronchograms visible. There is no evidence of pleural effusion or pneumothorax."
    report = generate_pneumonia_report(test_description)
    print("Generated Pneumonia Report:")
    print(report)

