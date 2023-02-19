import torch
from torch import nn
import torch.nn.functional as F

vocab_size=10000      # number of word in vocabulary
max_seq_length = 80   # maximum tweet length (length of tokenized tweet) (T)
d_emb = 100           # dimension of embedding vector
head_size = 20        # head size of self-attention 
num_heads = 6         # number of heads in multi-head self-attenton
n_layer = 3           # number of encoder blocks in model
dropout= 0.3          # dropout rate
num_classes = 3       # number of target classes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MainEmbadding(nn.Module):
    def __init__(self, vocab_size, d_emb, max_seq_length):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, d_emb) # (vocab_size,d_emb)
        self.pos_emb = nn.Embedding(max_seq_length, d_emb) # (T,d_emb)
        self.max_seq_length = max_seq_length
        
    def forward(self, x):
        emb1 = self.word_emb(x) # (B,T,d_emb)
        emb2 = self.pos_emb(torch.arange(self.max_seq_length, device="cpu")) # (T,d_emb)
        
        return emb1+emb2 #  (B,T,d_emb)
    
class Head(nn.Module):
    def __init__(self, head_size, d_emb, dropout=dropout):
        super().__init__()
        self.d_embed = d_emb # 20
        self.head_size = head_size # 16
        self.q = nn.Linear(d_emb, head_size, bias=False) # weight dims: (head_size, d_emb)
        self.k = nn.Linear(d_emb, head_size, bias=False) # weight dims: (head_size, d_emb)
        self.v = nn.Linear(d_emb, head_size, bias=False) # weight dims: (head_size, d_emb)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        query = self.q(x) # (B,T,head_size)
        key   = self.k(x) # (B,T,head_size)
        value = self.v(x) # (B,T,head_size)
        
        wei = torch.matmul(query, key.transpose(-1,-2)) * self.head_size**-0.5 # (B,T,head_size) @ (B,head_size,T) -> (B,T,T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        out = torch.matmul(wei, value) # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, d_emb, dropout=dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, d_emb) for _ in range(num_heads)]) # (B,T,head_size)
        self.proj = nn.Linear(head_size * num_heads, d_emb) # (B,T,d_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out # (B,T,d_emb)

class FeedFoward(nn.Module):
    def __init__(self, d_emb, dropout=dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_emb, 4 * d_emb),
            nn.ReLU(),
            nn.Linear(4 * d_emb, d_emb),
            nn.Dropout()
        )

    def forward(self, x):
        return self.net(x) # (B,T,d_emb)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, d_emb, num_heads):
        super().__init__()
        head_size = d_emb // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size, d_emb)
        self.ffwd = FeedFoward(d_emb)
        self.ln1 = nn.LayerNorm(d_emb)
        self.ln2 = nn.LayerNorm(d_emb)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x # (B,T,d_emb)

class FinalLayer(nn.Module):
    def __init__(self, d_emb, max_seq_length, num_classes):
        super().__init__()
        self.final_layer = nn.Linear(d_emb * max_seq_length, num_classes)
    
    def forward(self, x):
        x_flattened = torch.flatten(x, start_dim=1) # (B,T*C)
        unscaled_out = self.final_layer(x_flattened)
        probs = F.softmax(unscaled_out, dim=1)
        return probs # (B,num_classes)

class TransformerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embadding = MainEmbadding(vocab_size=vocab_size, d_emb=d_emb, max_seq_length=max_seq_length)
        self.blocks = nn.Sequential(*[Block(d_emb, num_heads=num_heads) for _ in range(n_layer)])
        self.final_layer = FinalLayer(d_emb=d_emb, max_seq_length=max_seq_length, num_classes=num_classes)
    
    def forward(self, x):
        x_emb = self.embadding(x)
        features = self.blocks(x_emb)
        probs = self.final_layer(features)
        return probs