from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F 


# ----------------------------------------------

class CausalSelfAttention(nn.Module):   # 因果(时序)attention: 掩盖掉 t 时刻后面的输入, 防止信息泄露
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0    # 如果 embedding 的维度 不能被 n个head 整除, 就抛出错误
        # Q, K, V
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)  # 3: query, key, value
        # output
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a "bias", more of a mask, but following the OpenAI/HuggingFace naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B,T,C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd, dim=2)

        # nh is "number of heads", hs, is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)  # (B, nh, T, hs)

        # attention (materializes the large (T,T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))   # attention 公式
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf"))   # masked, 防止信息泄露
        att = F.softmax(att, dim=-1)
        y = att @ v    # (B,nh,T,T) x (B,nh,T,hs) => (B,nh,T,hs), weighted sum operation

        # re-assemble all head outputs side by side, it is also a "Concatenation" operation
            # 通常: 在使用了 transpose, split(切片), view, narrow 这些操作之后, 需要重新整理数组元素, 否则可能发生意外错误(比如某些索引操作除错, 而且会导致性能下降)  
        y = y.transpose(1,2).contiguous().view(B,T,C)

        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    '''
    Multi-layer Perceptron.
    
    Notes:
        GELU (Gaussian Error Linear Unit), 在Pytorch中有两个版本, 其中原始的版本在 TensorFlow 中很慢而且似乎存在 numerical issue,
            所以以往基本上大家都用的是 近似 tanh 的形式, 故初始化时, 参数指定了 approximate="tanh"
            
            如今, 原版的问题似乎已经得到解决, 本repo的作者在后续的工作中是直接使用原版的(这里为了复原GPT-2, 就用了近似版)
    '''
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)   # fully connected, 升维, 隐空间空间维度为输入维度的4倍
        self.gelu   = nn.GELU(approximate="tanh")                   # 加非线性
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)   # fully connected, 降维, 回到原来维度
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    '''
    Attention Block.
    notes: 
        Attention layer is used for information exchange/collection, it is an aggregation function or somehow like a pooling function;
            it is also a weighted sum function, a reduce operation.
        
        MLP is like a mapping function, projecting modified tokens individually back into vocabulary space, and then decode some new words.
    '''

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)   # layer norm 1
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)   # layer norm 2
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))           # pre-nom + attention layer + residue connect
        x = x + self.mlp(self.ln_2(x))            # pre-norm + mlp (feat forward network) + residue connect
        return x

# 使用装饰器来构建数据类型, 好处是可以自动生成一些特殊方法, 如 __init__, __repr__, __eq__等, 无需手动编写, 初始化的时候也可以直接用类名
@dataclass 
class GPTConfig:
    block_size: int = 1024           # max input sequence length
    vocab_size: int = 50257          # vocabulary size
                                        # 50,000 BPE merges + 256 bytes tokens (the leave of the BPE tree) + 1 (the "<|endoftext|>" token)
    n_layer: int = 12                # number of transformer layer
    n_head: int = 12                 # number of attention head
    n_embd: int = 768                # embedding dimension

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()          # init the base class, ie. nn.Module constructor
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # weight token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),  # weight position embedding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # Causal self attention layer, 
                                                                                # 使用 ModuleList 就可以用 ".数字" 来索引每一个 head, 如 transformer.h.0.c_attn.weight
            ln_f = nn.LayerNorm(config.n_embd),                    # layer norm final 
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # final module head, ie. the language model head (for down-stream task)


    def forward(self, idx):
        # idx is of shape (B, T), T is short for "Time" 
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        # forward token
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape: (T)
        pos_emb = self.transformer.wpe(pos)  # position embedding (T, n_embd),  实际上是 (1, T, n_embd)
        tok_emd = self.transformer.wte(idx)  # token embedding (B, T, n_embd)
        x = tok_emd + pos_emb                # 隐含一个broadcasting操作
        
        # forward blocks(self attention) of the transformer
        for block in self.transformer.h:
            x = block(x)
        
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logit = self.lm_head(x)  # (B,T,vocab_size)
        
        return logit

    # -------------- 加载 OpenAI 的预训练权重 到上面手撸的 GPT-2 -----
    @classmethod
    def from_pretrained(cls, model_type):
        """
        Load pretrained GPT-2 model weights from huggingface.
        """
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)
        
        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)   # 使用面用装饰器"@dataclass"定义过的数据类型创建实例, 调用的依旧是 __init__ 构造函数, 只不过前面因为用了装饰器, 无需手写 __init__
        model = GPT(config)
        sd = model.state_dict()             # sd: state dict, 这里返回的是"模型参数字典的引用", 如果修改了sd的值, 模型权重也会对应被修改!
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        
        # 因为 OpenAI 实现的 GPT-2 是用 TensorFlow 写的, 它的 Conv1D 模块对应的权重矩阵跟 pytorch 的实现恰好是 transpose 关系, 所以这里需要特殊处理一下
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        # 复制权重到手撸的GPT-2模型中
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():                         ## 只复制权重矩阵, 不复制计算图
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():                         ## 只复制权重矩阵, 不复制计算图
                    sd[k].copy_(sd_hf[k])

        return model

# ----------------------------- test of our GPT-2 -----------------------------------------
device = "cpu"
# ning: 用2080Ti做实验, 指定一下GPU
device = torch.device("cuda:2")   # 用2080Ti做实验
print( f"we are using: {torch.cuda.get_device_name(2)}")

# 或者自动检测GPU, CPU
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"    # macbook
else:
    device = "cpu"

model = GPT.from_pretrained("gpt2")
# print("ohhhhhhhhhhhhhhh, pretrained weights load success!!!")   

num_return_sequences = 5
max_length = 30
model.to(device)
model.eval()

# get prefix tokens from raw text
import tiktoken
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I am a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (8) -> (1,8) -> (5,8)
x = tokens.to(device)

# Generate!! 
# B = 5, T = 8;
#   B为batch, 因为这里想针对 "Hello, I am a language model," 这段话, 生成5个版本的后续输出
torch.manual_seed(42)         # CPU 的随机种子
torch.cuda.manual_seed(42)    # GPU 的随机种子
while x.size(1) < max_length:  # 每次预测出来的词加入到 pre-context 中, 当长度小于1024时才继续往下生成内容
    # get output logit
    with torch.no_grad():
        logits = model(x)   # (B, T, vocab_size)
        
        # take the logits at the last position  (最后一个才是prediction)
        logits = logits[:,-1,:]  #  (B, vocab_size)
        
        # get the probabilities
        probs = F.softmax(logits, dim = -1) # (B, vocab_size)
        
        # probs 相当于单词表中下一个单词可能出现的概率, 下面我们取可能性最高的前50个词对应的index
            # topk 会只保留最大的前50个概率, 让后将其他元素置零, 这样就可以避免采样到 很不常见的单词  
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5,50), topk_indices is (5,50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        
        # 把前50个候选词的概率作为数值, 丢到"多项式分布"公式中, 从而决定下一个词
            # 这样做的意义是: 将50个词的概率作为权重, 让模型在预测下一个词的时候具有更丰富的多样性, 而不是直接依赖模型给出的最可能的下一词
        # select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, 1)    # (B,1)
        
        # 对B个版本的预测分别获取各自的下一个词在词表中的index
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B,1)
        
        x = torch.cat((x, xcol), dim=1)
        

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decode = enc.decode(tokens)
    print("->", decode)

