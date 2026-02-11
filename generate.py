def generate(model, tokenizer, start_text, length=20):
    model.eval()
    tokens = tokenizer.encode(start_text)
    input_tensor = torch.tensor(tokens).unsqueeze(0)

    for _ in range(length):
        output = model(input_tensor)
        next_token = torch.argmax(output[0, -1]).item()
        tokens.append(next_token)
        input_tensor = torch.tensor(tokens).unsqueeze(0)

    return tokenizer.decode(tokens)
