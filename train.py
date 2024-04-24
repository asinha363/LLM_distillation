import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM, AutoTokenizer

# Configuration
class Config:
    teacher_model_name = 'allenai/llama-13b'
    student_model_config = 'llama-tiny'
    num_epochs = 3
    batch_size = 4
    learning_rate = 5e-5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize models
def initialize_models(config):
    teacher_model = LlamaForCausalLM.from_pretrained(config.teacher_model_name)
    student_model = LlamaForCausalLM.from_pretrained(config.student_model_config)
    return teacher_model.to(config.device), student_model.to(config.device)

# Data preparation
def get_dataloader(batch_size):
    tokenizer = AutoTokenizer.from_pretrained(Config.teacher_model_name)
    texts = ["Hello, world!", "How are you?", "Knowledge distillation example."]  # example texts
    encodings = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    dataset = torch.utils.data.TensorDataset(encodings.input_ids, encodings.attention_mask)
    return DataLoader(dataset, batch_size=batch_size)

# Training loop
def train_model(teacher_model, student_model, dataloader, config):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(student_model.parameters(), lr=config.learning_rate)

    teacher_model.eval()
    student_model.train()

    for epoch in range(config.num_epochs):
        for batch in dataloader:
            inputs, masks = batch
            inputs, masks = inputs.to(config.device), masks.to(config.device)

            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=inputs, attention_mask=masks).logits

            student_outputs = student_model(input_ids=inputs, attention_mask=masks).logits

            loss = criterion(student_outputs, teacher_outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Save the model
def save_model(model, path='student_model.pt'):
    torch.save(model.state_dict(), path)

# Main function
def main():
    config = Config()
    teacher_model, student_model = initialize_models(config)
    dataloader = get_dataloader(config.batch_size)
    train_model(teacher_model, student_model, dataloader, config)
    save_model(student_model)

if __name__ == '__main__':
    main()
