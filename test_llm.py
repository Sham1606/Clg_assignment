import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json

# Load the fine-tuned model and tokenizer
model_path = "./gpt2-medical-advanced-finetuned"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Make sure the model is in evaluation mode
model.eval()


def generate_text(prompt, max_length=200):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, temperature=0.7, top_k=50,
                            top_p=0.95)
    return tokenizer.decode(output[0], skip_special_tokens=False)


def test_generation():
    print("Testing text generation...")
    prompts = [
        "<image>Chest X-ray, PA view\n<description>The chest X-ray shows bilateral patchy opacities in the lower lobes.",
        "<image>Chest CT scan, axial view\n<description>CT scan reveals a solitary pulmonary nodule in the right upper lobe, measuring 2.5 cm in diameter.",
        "<image>Chest MRI, coronal view\n<description>MRI demonstrates a large mass in the left hemithorax with invasion of the chest wall."
    ]

    for prompt in prompts:
        generated_text = generate_text(prompt)
        print(f"Prompt: {prompt}")
        print(f"Generated text: {generated_text}\n")


def test_special_tokens():
    print("Testing special tokens...")
    special_tokens = ['<image>', '<description>', '<analysis>', '<recommendation>']
    for token in special_tokens:
        if token in tokenizer.get_vocab():
            print(f"Special token '{token}' is present in the vocabulary.")
        else:
            print(f"Special token '{token}' is NOT present in the vocabulary.")


def test_model_output_structure():
    print("Testing model output structure...")
    prompt = "<image>Chest X-ray, PA view\n<description>Normal chest X-ray."
    generated_text = generate_text(prompt)

    expected_sections = ['<image>', '<description>', '<analysis>', '<recommendation>']
    for section in expected_sections:
        if section in generated_text:
            print(f"Section '{section}' is present in the generated text.")
        else:
            print(f"Section '{section}' is NOT present in the generated text.")


def test_consistency():
    print("Testing consistency of generated reports...")
    prompt = "<image>Chest X-ray, PA view\n<description>The chest X-ray shows clear lung fields with no evidence of consolidation or effusion."

    reports = [generate_text(prompt) for _ in range(5)]

    # Compare the analysis sections
    analyses = [report.split('<analysis>')[1].split('<recommendation>')[0].strip() for report in reports]

    consistent = all(analysis == analyses[0] for analysis in analyses)
    if consistent:
        print("The model generates consistent analyses for the same input.")
    else:
        print("The model generates varying analyses for the same input.")
        for i, analysis in enumerate(analyses):
            print(f"Analysis {i + 1}: {analysis}")


def main():
    print("Starting LLM model tests...\n")

    test_generation()
    print("\n" + "=" * 50 + "\n")

    test_special_tokens()
    print("\n" + "=" * 50 + "\n")

    test_model_output_structure()
    print("\n" + "=" * 50 + "\n")

    test_consistency()
    print("\n" + "=" * 50 + "\n")

    print("LLM model tests completed.")


if __name__ == "__main__":
    main()

