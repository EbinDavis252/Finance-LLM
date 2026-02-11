import torch
import torch.nn as nn

class MiniFinanceLLM(nn.Module):
    def __init__(self, vocab_size, embed_size=32, heads=2, layers=1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(100, embed_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=heads,
            dropout=0.2,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.fc = nn.Linear(embed_size, vocab_size)
    
    def forward(self, x):
        positions = torch.arange(0, x.size(1)).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(positions)
        x = self.transformer(x)
        x = self.fc(x[:, -1, :])
        return x
