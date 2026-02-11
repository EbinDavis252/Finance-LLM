import torch
import torch.nn as nn
import torch.optim as optim

def train(model, data, epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data[:-1])
        loss = criterion(output.view(-1, output.size(-1)), data[1:].view(-1))
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
