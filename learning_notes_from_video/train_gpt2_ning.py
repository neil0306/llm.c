from dataclasses import dataclass
import math
import inspect
import os
import torch
import torch.nn as nn
from torch.nn import functional as F 


# ----------------------------------- Define our GPT-2 model ------------------------------------

class CausalSelfAttention(nn.Module):   # 因果(时序)attention: 掩盖掉 t 时刻后面的输入, 防止信息泄露
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0    # 如果 embedding 的维度 不能被 n个head 整除, 就抛出错误
        # Q, K, V
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)  # 3: query, key, value
        # output
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1      # 由于走到这部分之后就到 residue connection 的加法, 为了确保数值的 std 仍然是在1附近, 这里需要一个scale down 的flag
        
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a "bias", more of a mask, but following the OpenAI/HuggingFace naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))  # 下三角阵, (1,1,1024,1024)

    def forward(self, x):
        B,T,C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch
        qkv = self.c_attn(x)   # (B,T, 3*n_embd), 严格来说, q,k,v 在生成的时候分别对应着 w_q, w_k, w_v 的, 这里只是将这三个矩阵叠起来了, 所以可以直接用一层比较大的 Linear 完成.
        q,k,v = qkv.split(self.n_embd, dim=2) # (B,T, n_embd), (B,T, n_embd), (B,T, n_embd)

        # nh is "number of heads", hs, is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)  # (B, nh, T, hs)

        # ------- original causal self-attention -------
        # attention (materializes the large (T,T) matrix for all the queries and keys)
        # att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))   # attention 公式
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf"))   # masked, 只保留下三角阵的数值(按照时间顺序, q的每一行只比上一行多看到一个k的信息), 防止信息泄露
        # att = F.softmax(att, dim=-1)
        # y = att @ v    # (B,nh,T,T) x (B,nh,T,hs) => (B,nh,T,hs), weighted sum operation
        
        #  ------------- Flash Attention -----------------
        # 由于 torch.compile 无法将 Attention 操作进行识别并整合(无法自动实行 kernel fusion), 故这里需要引入 Flash Attention
        # 它整合了 matmul(矩阵乘法, q@k), Dropout, Softmax, mask 和 matmul(矩阵乘法, att@v), 这样就避免了将 (N,N) 大小的 Attention matrix 在 HBM 中反复读写造成的延迟 
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

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
        self.c_proj.NANOGPT_SCALE_INIT = 1      # 由于走到这部分之后就到 residue connection 的加法, 为了确保数值的 std 仍然是在1附近, 
                                                # 这加一个 scale down 的flag, 方便在GPT类里初始化权重的时候做特殊处理

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

        # 注意: 在Attention is all you need 以及 GPT-2 源码(tensorflow版本)中, 模型最开始的 token embedding 和 最后输出阶段生成 logits 的时候, 使用的权重矩阵都是相同的, 文章中称为 weight sharing
        # 在上面的实现中, 我们其实还没有进行 weight sharing 的操作, 故需要进行如下改动
        self.transformer.wte.weight = self.lm_head.weight   # 由于是共享权重, 所以在初始化阶段其实进行了2次初始化, 不过影响不大
        
        # 用指定的方式初始化参数
        self.apply(self._init_weights)   # apply() 方法是从 nn.Module 中集成过来的, 它负责初始化所有的子模块
    
    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):  # 初始化模型中所有 linear 层
            if hasattr(module, "NANOGPT_SCALE_INIT"):          # 前面的子模块中增加了这个flag, 这里检查一下flag, 然后做针对性的初始化
                std *= (2 * self.config.n_layer) ** -0.5            # 在GPT-2的paper中(介绍不同大小的模型的表格下面的段落)有提到, 为了确保residue stream的输出依旧保持方差为1, 需要对权重进行scale down
                                                                    # 这里的倍数 2 是因为 Block 模块(self-attention) 的 forward path 里面, Attention 和 MLP 都有一次残差链接
            
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)  # 均值为0.0 和 std为0.2 是从 OpenAI 源码中抄来的, 为什么这么用原因未知
                                                                        # 如果按照Xavier初始化方法, 方差 = 1/本层输入元素个数, 故标准差 std = 1/sqrt(n)
                                                                        # 本层输入元素个数其实就是模型的 d_model, 对于124M模型, d_model = 768, std应为 0.036; 如果是1600, std 就差不多是 0.025
                                                                        # 故, 猜测这是根据 Xavier 初始化得到的数值
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)                     # pytorch 里, bias的默认初始化并不是0, 而是按照均匀分布进行初始化
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)   # 同上

    def configure_optimizer(self, weight_decay, learning_rate, device):
        """
        GPT-3: "All models use weight decay of 0.1 to provide a small amount of regularization."
        
        1. 取出所有需要计算梯度的模型参数
        2. 只对2D参数做 weight decay regularization, 因为1D的参数要么是scale factor, 要么是layernorm, 或者linear layer的bias, 这些参数不进行decay也OK的
        """
        
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}     # 获取模型参数
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}  # 筛选需要计算梯度的参数
        
        # create optim groups, any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay. all biases and layer norms don't.
        decay_params = [p for n,p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]
        
        num_decay_params = sum(p.numel() for p in  decay_params)
        num_nodecay_params = sum(p.numel() for p in  nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params} parameters")

        # 早期版本的pytorch中, adamw内并没有 fused 操作(也就是 kernel fused), 所以这里进行一下检查
        # create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters  # 检查pytorch是否支持 fused 加速 (带签名就是支持)
        use_fused = fused_available and 'cuda' in device                              # 如果支持, 并且用的是 N卡, 则弄一个flag给AdamW
        print(f"using fused Adamw: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

        return optimizer


    def forward(self, idx, targets=None):
        # idx is of shape (B, T), T is short for "Time" 
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        # forward token
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # 生成位置编号, shape: (T)
        pos_emb = self.transformer.wpe(pos)  # position embedding (T, n_embd),  实际上是 (1, T, n_embd)
        tok_emd = self.transformer.wte(idx)  # token embedding (B, T, n_embd)
        x = tok_emd + pos_emb                # 隐含一个broadcasting操作
        
        # forward blocks(self attention) of the transformer
        for block in self.transformer.h:
            x = block(x)
        
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)
        
        loss = None
        if(targets is not None):
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))  # cross-entrpy只能接受2维向量, logits view 之后shape变成 (B*T, vocab_size), target 变成(b*T)
        return logits, loss

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


# ----------------------------- Stage-5: Define our DataLoader (with DDP setting options) -------------------------------------------------------------------------------
import tiktoken
import numpy as np

def load_tokens(filename):
    """
    由于作者的最终目的是用C语言跑模型, 故他将下载的数据写入了一个.bin 文件中, 因此, 要注意读取数据无法直接用 numpy 来读.
    """
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)                        # 前4个字节是作者在写入bin文件的时候塞进去的校验信息, np.frombuffer 将读取的东西转成 numpy 数组
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"  # 这个数是写bin文件的时候自定义的
        assert header[1] == 1, "unsupported version"                                 # 也是写bin文件的时候自定义的
        ntok = header[2] # number of tokens (claimed)                                # 写bin文件的时候对文件内的token总数进行了记录, 这里读回来
        # the rest of it are tokens, stored as uint16
        buf = np.frombuffer(f.read(), dtype=np.uint16)                               # 读取所有token
    assert len(buf) == ntok, "number of tokens read does not match header?"
    
    tokens = torch.tensor(buf.astype(np.int32), dtype=torch.long)                    # bin文件中的token都是 uint16 类型(省空间), 这里进行类型转换
    return tokens

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank    # ddp 参数
        self.num_processes = num_processes  # ddp 参数
        assert split in {"train", "val"}

        # ---- In pre-version (Stage 1~4), we use tinyshakespeare dataset, but now we want to use the splited fineweb10B dataset --
        # with open("dev/data/tinyshakespeare/tiny_shakespeare.txt", 'r') as f:
        #     text = f.read()
        # enc = tiktoken.get_encoding("gpt2")
        # tokens = enc.encode(text)
        # self.tokens = torch.tensor(tokens)
        # if master_process:
        #     print(f"Loaded {len(tokens)} tokens.")
        #     print(f"1 epoch = {len(self.tokens) // (B*T)} batches")
        # self.current_position = self.B * self.T * self.process_rank  # 从原来的0, 改成通过 GPU 编号计算state
        # -------------------------------------------------------------------------------------------------------------
        
        # get the shard filenames
        data_root = "dev/data/fineweb10B"
        shards = os.listdir(data_root)               # 遍历文件夹, 获取文件名
        shards = [s for s in shards if split in s]   # split用来标识 train 或 val
        shards = sorted(shards)                      # 按文件名从小到大排序
        shards = [os.path.join(data_root, s) for s in shards]  # 完整的文件路径(相对路径)
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        
        # 初始化数据读取的状态
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = (buf[:-1]).view(B,T) # input
        y = (buf[1:]).view(B,T)  # ground truth
        
        # advance the position in the tensor
        self.current_position += B*T * self.num_processes            # 改成ddp配置: 从原来的 B*T 改成 "B*T * self.num_processes", 通过GPU总数计算出下一次整体读取的起始位置
        
        # ----- stage 1~4: the code of using tinyshakespeare dataset  ------------
        # if loading the next batch would be out of bounds, reset
        # if self.current_position + (B*T * self.num_processes +1) > len(self.tokens):   # 改成ddp配置
        #     self.current_position = 0
        # ------------------------------------------------------------------------
        
        # ----- stage 5: the code if using fineweb10B dataset ----------
        # if loading the next batch would be out of bound, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):  # "self.current_position + (B * T * self.num_processes + 1)" 表示下一次读取数据的起始位置
            self.current_shard = (self.current_shard + 1) % len(self.shards)             # 取模, 可以确保shard的数值不会超过数据集中shard的总数, 并且只要超过, 就会从第一个shard开始重新读取数据
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank                            # 计算当前GPU应该读shard里的哪一段数据
        return x,y

# ----------------------------- Stage-1: test of our GPT-2 (with/without loading pre-trained weights) -----------------------------------------
# device = "cpu"
# # ning: 用2080Ti做实验, 指定一下GPU
# # device = torch.device("cuda:2")   # 用2080Ti做实验
# # print( f"we are using: {torch.cuda.get_device_name(2)}")

# # 或者自动检测GPU, CPU
# if torch.cuda.is_available():
#     device = "cuda"
# elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#     device = "mps"    # macbook
# else:
#     device = "cpu"

# # model = GPT.from_pretrained("gpt2")      # 加载 OpenAI 提供的预训练权重
# # print("ohhhhhhhhhhhhhhh, pretrained weights load success!!!")   
# model = GPT(GPTConfig)  # 不加载预训练参数, 直接用pytorch自带的随机初始化

# num_return_sequences = 5
# max_length = 30
# model.to(device)
# model.eval()

# # ------------- Generate!! --------------------- 
# # get prefix tokens from raw text
# import tiktoken
# enc = tiktoken.get_encoding("gpt2")
# tokens = enc.encode("Hello, I am a language model,")
# tokens = torch.tensor(tokens, dtype=torch.long) # (8)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (8) -> (1,8) -> (5,8)
# x = tokens.to(device)

# # B = 5, T = 8;
# #   B为batch, 因为这里想针对 "Hello, I am a language model," 这段话, 生成5个版本的后续输出
# torch.manual_seed(42)         # CPU 的随机种子
# torch.cuda.manual_seed(42)    # GPU 的随机种子
# while x.size(1) < max_length:  # 每次预测出来的词加入到 pre-context 中, 当长度小于1024时才继续往下生成内容
#     # get output logit
#     with torch.no_grad():
#         logits = model(x)   # (B, T, vocab_size)
        
#         # take the logits at the last position  (最后一个才是prediction)
#         logits = logits[:,-1,:]  #  (B, vocab_size)
        
#         # get the probabilities
#         probs = F.softmax(logits, dim = -1) # (B, vocab_size)
        
#         # probs 相当于单词表中下一个单词可能出现的概率, 下面我们取可能性最高的前50个词对应的index
#             # topk 会只保留最大的前50个概率, 让后将其他元素置零, 这样就可以避免采样到 很不常见的单词  
#         # do top-k sampling of 50 (huggingface pipeline default)
#         # topk_probs here becomes (5,50), topk_indices is (5,50)
#         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        
#         # 把前50个候选词的概率作为数值, 丢到"多项式分布"公式中, 从而决定下一个词
#             # 这样做的意义是: 将50个词的概率作为权重, 让模型在预测下一个词的时候具有更丰富的多样性, 而不是直接依赖模型给出的最可能的下一词
#         # select a token from the top-k probabilities
#         ix = torch.multinomial(topk_probs, 1)    # (B,1)
        
#         # 对B个版本的预测分别获取各自的下一个词在词表中的index
#         # gather the corresponding indices
#         xcol = torch.gather(topk_indices, -1, ix) # (B,1)
        
#         x = torch.cat((x, xcol), dim=1)
        

# # print the generated text
# for i in range(num_return_sequences):
#     tokens = x[i, :max_length].tolist()
#     decode = enc.decode(tokens)
#     print("->", decode)

# ---------------------------------- Stage-2: train our GPT-2  ---------------------------------------------------
# device = "cpu"
# if torch.cuda.is_available():
#     device = "cuda"
# elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#     device = "mps"    # macbook
# else:
#     device = "cpu"

# print(f"We are using {device} ...")

# # 固定随机种子, 便于复现结果
# torch.manual_seed(1337)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(1337)
# elif torch.backends.mps.is_available():
#     torch.mps.manual_seed(1337)

# # get a data batch 
# import tiktoken
# enc = tiktoken.get_encoding("gpt2")
# with open("dev/data/tinyshakespeare/tiny_shakespeare.txt", 'r') as f:
#     text = f.read()
# text = text[:1000]
# tokens = enc.encode(text)  # 1000个单词进行预处理后剩下 285 个token
# B,T, = 4, 32
# buf = torch.tensor(tokens[:B*T + 1])
# buf = buf.to(device)          # 先把buffer丢到GPU, 后续的x,y就不用再次操作了
# x = buf[:-1].view(B,T)  # (B,T)
# y = buf[1:].view(B,T)   # (B,T)

# # get logits
# model = GPT(GPTConfig)
# model.to(device)
# # model = torch.compile(model)  # pytorch 2.0 之后支持模型编译, 类似使用 GCC/G++ 之类的编译器编译代码, 而不是直接用python解释器去跑
#                                 # mac上的pytorch目前不支持编译
                                
#                                 # torch.compile() 的功能主要是过一遍整个网络, 然后将一些操作进行整合, 比如 GELU 激活函数, 里面有一些求指数, 开方, 乘加操作, 
#                                 # 由于这些操作都是对同一拨数据按顺序计算, compile的时候将它们直接整合(称为kernel fusion), 这样就可以避免多次数据的存取, 降低延迟, 
#                                 # 因此 torch.compile 是加速模型的一种通用手段

# # logits, loss = model(x, y)  # 输出的loss差不多是10.9930(或者11左右), 
# #                             # 注意现在还没有开始训练, 输出这个数值是因为 cross-entropy 本质上就是计算 -ln(probability),
# #                             # 由于我们词表大小是 20257, 如果初始化的模型等同于均匀分布, 那么我们预测的下一个词的概率就应该接近 1/20257, 此时得到 9.91625, 
# #                             # 因此这里输出 10.9930 是可以接受的初始化状态
# # print(loss)

# # optimizer!!
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)   # adamw 可以当做是 adam 优化器修了一个bug;  
#                                                              # 3e-4 是大家常用的 early age debug Learning Rate

# # # debug stage: 用小批量数据(一直不加新数据进来), 反复更新梯度, 看看模型是否能过拟合, 如果会过拟合, 证明模型在正常训练
# # for i in range(50):
# #     optimizer.zero_grad()     # 一定以及 清空历史 梯度!!!
# #     logits, loss = model(x, y)
# #     loss.backward()    # 计算梯度
# #     optimizer.step()   # 更新参数
# #     print(f"step {i}, loss: {loss.item()}")   # loss.item() 可以将tensor换成为 float, 并把数据放回CPU


# # train model
# import time
# train_loader = DataLoaderLite(B=4, T=32)  # batch size 尽可能使用2的倍数, 因为硬件都是2进制, 这样可以让机器运行效率高一些

# # torch.set_float32_matmul_precision("high")  # hight: 做乘法的时候使用TF32(精度下降), highest: 做乘法的时候一直使用FP32
#                                                 # 只在 A100之后 的N卡上有用, 在mac上无效

# for i in range(50):
#     t0 = time.time()
#     x, y = train_loader.next_batch()
#     x, y = x.to(device), y.to(device)
#     optimizer.zero_grad()     # 一定以及 清空历史 梯度!!!
    
#     # --------- 使用混精度数据类型加速 ------------
#     # with torch.autocast(device_type=device, dtype=torch.bfloat16):  # mac不支持, 只有安培架构(30系列显卡)之后才支持
#     #     logits, loss = model(x, y)
#     logits, loss = model(x, y)   # 非混精度模式
    
    
#     # import code; code.interact(local=locals())   # 通过这行代码, 我们可以在终端触发一个 interactive console, 直接进行一些debug操作
    
#     loss.backward()    # 计算梯度
    
#     norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # GPT-3里提到的操作: 将 global norm of the gradient 给 clip 到 1.0
#                                                                     # 这个函数做的操作: 将所有参数的梯度求平方和, 再开方 (求L2范数); 如果范数的值大于指定的数值, 就会缩放这些梯度来满足指定的范数大小条件; 返回的norm是裁剪之前的梯度范数值.
#                                                                     # 作用: 防止某些batch产生的loss很大, 导致梯度发生剧烈变化, 让训练不稳定
#                                                                     # 通常我们会把这个函数返回的 norm 值打印出来, 如果这个norm一直都挺平稳的, 不会随着训练一直增大, 在一定程度上说明模型正在稳定训练
    
#     optimizer.step()   # 更新参数
    
#     # torch.cuda.synchronize()   # wait for GPU to finish work (只有在N卡上有用, mac上无效)
    
#     t1 = time.time()
#     dt = (t1 - t0) * 1000 # time difference in miliseconds
#     tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
#     print(f"step {i} | loss: {loss.item():.6f} | norm: {norm:.4f} | dt: {dt:2f}ms | tok/sec: {tokens_per_sec}")   # loss.item() 可以将tensor换成为 float, 并把数据放回CPU

# import sys; sys.exit(0)   # 代码走到这里就会停止, 这是一个debug的时候比较不错的方式

# --------------------------------------------------------------------------------------------------------------------------------------------

# # -------------------- Stage-3: train GPT-2 following GPT-3 paper's setup detail ----------------
# import time
# import tiktoken

# device = "cpu"
# if torch.cuda.is_available():
#     device = "cuda"
# elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#     device = "mps"    # macbook
# else:
#     device = "cpu"

# print(f"We are using {device} ...")

# # 固定随机种子, 便于复现结果
# torch.manual_seed(1337)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(1337)
# elif torch.backends.mps.is_available():
#     torch.mps.manual_seed(1337)

# model = GPT(GPTConfig)
# model.to(device)
# # model = torch.compile(model)  # pytorch 2.0 之后支持模型编译, 类似使用 GCC/G++ 之类的编译器编译代码, 而不是直接用python解释器去跑
#                                 # mac上的pytorch目前不支持编译
                                
#                                 # torch.compile() 的功能主要是过一遍整个网络, 然后将一些操作进行整合, 比如 GELU 激活函数, 里面有一些求指数, 开方, 乘加操作, 
#                                 # 由于这些操作都是对同一拨数据按顺序计算, compile的时候将它们直接整合(称为kernel fusion), 这样就可以避免多次数据的存取, 降低延迟, 
#                                 # 因此 torch.compile 是加速模型的一种通用手段

# # optimizer!!
# # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)   # adamw 可以当做是 adam 优化器修了一个bug;  
#                                                                 # 3e-4 是大家常用的 early age debug Learning Rate
# optimizer = model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)  # 这里的 weight_decay 由模型内的一个自定义方法来完成

# # ------- cosine decay implementation ----------
# max_lr = 6e-4           # GPT-3里的参数
# min_lr = max_lr * 0.1
# warmup_steps = 10
# max_steps = 50          # 最大循环次数, 为了方便演示, 只设置50次

# def get_lr(step):
#     # 1) linear warmup for warmup_iters steps (按照 step 线性增涨)
#     if step < warmup_steps:
#         return max_lr * (step+1) / warmup_steps     # 加1是因为 lr 设置为0没啥意义
    
#     # 2) if step > lr_decay_iters, return min learning rate
#     if step > max_steps:
#         return min_lr
    
#     # 3) in between, use consine decay down to min learning rate
#     decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
#     assert 0 <= decay_ratio <= 1
#     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff start at 1.0 and goes to 0
#     return min_lr + coeff * (max_lr - min_lr)

# # train model
# '''
# 在 GPT-3 paper 的模型超参数表格中, 参数量约为125M的模型对应的 batch size 为 0.5M(这是每次丢进模型的token总数量), lr 为 6e-4
#     因为 batch size 与 lr, AdamW 的 betas 等超参数是存在关联性的, 因此, 如果我们其他地方已经是用了GPT-3的超参数, 那么每次丢进模型的token数量也应该尽可能达到相应的数量.
#         这个方法就是 按照GPU的显存, 设置一个最大的batch size, 但是先不更新参数, 而是将梯度累加, 当 feed 进模型的token数量达到我们想要的数值时才进行参数更新.
# '''
# total_batch_size = 524288 # 0.5M token 这个数不是2的倍数, 这里取 2^19=524288, 某种程度上或许可以让GPU的利用率高些
# B = 4   # micro batch
# T = 1024  # sequence length (注意GPT-2单次最长只支持1024个token)
# assert total_batch_size % (B*T) == 0, "Please make sure total_batch_size is divisible by B*T!"
# grad_accum_steps = total_batch_size // (B*T)   # 计算要走多少个 step 之后才更新参数
# print(f"total desired batch size: {total_batch_size}")
# print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# train_loader = DataLoaderLite(B=B, T=T)  # batch size 尽可能使用2的倍数, 因为硬件都是2进制, 这样可以让机器运行效率高一些

# # torch.set_float32_matmul_precision("high")  # hight: 做乘法的时候使用TF32(精度下降), highest: 做乘法的时候一直使用FP32
#                                                 # 只在 A100之后 的N卡上有用, 在mac上无效

# loss_accum = 0.0
# for step in range(max_steps):
#     t0 = time.time()

#     optimizer.zero_grad()     # 一定以及 清空历史 梯度!!!
    
#     for micro_step in range(grad_accum_steps):    # 为了使丢进模型的 total batch token 能与 GPT-3 paper 里的超参数匹配, 这里需要使用 accumulate gradient 的技巧(也就是这个内层for循环)
#         x, y = train_loader.next_batch()
#         x, y = x.to(device), y.to(device)
#         # --------- 使用混精度数据类型加速(只有 30系列之后 的 N卡 支持) ------------
#         # with torch.autocast(device_type=device, dtype=torch.bfloat16):  # mac不支持, 只有安培架构(30系列显卡)之后才支持
#         #     logits, loss = model(x, y)
#         logits, loss = model(x, y)   # 非混精度模式
#         loss = loss / grad_accum_steps                # 由于这里直接进行了梯度累加, 而计算 loss 的时候, 函数内部是自动求了平均的, 因此这里需要额外除以 grad_accum_steps 进行修正 (本质上是一个乘法分配律) -- 在 当前文件夹下的 play.ipynb 中有代码示例
#         loss_accum += loss.detach()                   # 只是为了打印出来看一看
#         loss.backward()    # 计算梯度 (在当前这个嵌套的for循环里, 基于pytorch的特性, 梯度会自动进行累加)
    
#     norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # GPT-3里提到的操作: 将 global norm of the gradient 给 clip 到 1.0
#                                                                     # 这个函数做的操作: 将所有参数的梯度求平方和, 再开方 (求L2范数); 如果范数的值大于指定的数值, 就会缩放这些梯度来满足指定的范数大小条件; 返回的norm是裁剪之前的梯度范数值.
#                                                                     # 作用: 防止某些batch产生的loss很大, 导致梯度发生剧烈变化, 让训练不稳定
#                                                                     # 通常我们会把这个函数返回的 norm 值打印出来, 如果这个norm一直都挺平稳的, 不会随着训练一直增大, 在一定程度上说明模型正在稳定训练
    
#     # 按照GPT-3文章的描述, lr 按照 consine decay 的方式进行衰减
#     lr = get_lr(step)                                               # 按照 cosine decay 的方式计算 当前的lr
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = lr
    
#     optimizer.step()   # 更新参数
    
#     # torch.cuda.synchronize()   # wait for GPU to finish work (只在 N卡 上有用, mac上无效)
    
#     t1 = time.time()
#     dt = (t1 - t0)  # time difference in seconds
#     tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
#     tokens_per_sec = tokens_processed / dt
#     print(f"step {step} | loss: {loss_accum.item():.6f} | lr: {lr:.5f} | norm: {norm:.4f} | dt: {dt:2f} s | tok/sec: {tokens_per_sec:.2f}")   # loss.item() 可以将tensor换成为 float, 并把数据放回CPU

# import sys; sys.exit(0)   # 代码走到这里就会停止, 这是一个debug的时候比较不错的方式


# -------------------------- Stage-4: train GPT-2, PPD 版本 -------------------------------
# import time
# import tiktoken

# device = "cpu"
# if torch.cuda.is_available():
#     device = "cuda"
# elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#     device = "mps"    # macbook
# else:
#     device = "cpu"


# import torch.distributed as dist   # DDP 操作的核心模块, 里面包含各种 ddp function
# from torch.distributed import init_process_group, destroy_process_group
# from torch.nn.parallel import DistributedDataParallel as DDP   # DDP 模块

# # set up DDP (distributed data parallel).
# # torchrun sets this env variable
# ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?  (对于 torchrun 的启动方式来说, 这并不算一种好的检测方式) 
#                                             # 这里指定nccl作为后端，nccl是NVIDIA开发的专门用于GPU集群通信的库，能够优化GPU间的通信效率。
# if ddp:
#     # use of DDP atm demands CUDA, we set the device appropriately according to rank
#     assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
#     init_process_group(backend='nccl')
#     ddp_rank = int(os.environ['RANK'])              # ddp_rank 通常表示当前进程在所有分布式进程中的编号, 如果是单机8卡环境, ddp_rank 数值范围就是 0~7
#     ddp_local_rank = int(os.environ['LOCAL_RANK'])  # LOCAL_RANK 表示获取当机器的上局部标识, 也就是当前机器上的GPU ID(rank of the GPU on a single node)
#     ddp_world_size = int(os.environ['WORLD_SIZE'])  # world_size 表示所有分布式节点包含的进程数(number of processes), 由于一个GPU对应一个进程, 这个也可以理解为GPU总数
#     device = f'cuda:{ddp_local_rank}'
#     torch.cuda.set_device(device)
#     master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.   (可以理解为主进程, 表示这个进程需要 兼顾 记录log, 记录checkpoint之类的工作)
# else:
#     # vanilla, non-DDP
#     ddp_rank = 0         # 全局进程ID, 这表示在整个分布式计算环境中, 当前运行的进程ID为0 (因为是单机单卡, 进程就一个, 编号为0)
#     ddp_local_rank = 0   # 使用机器上编号为0的GPU
#     ddp_world_size = 1   # 单机单卡
#     master_process = True # 表示当前进程为主进程

#     # attempt to autodetect the device
#     device = "cpu"
#     if torch.cuda.is_available():
#         device = "cuda"
#     elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#             device = "mps"
#     print(f"using device: {device}")

# # 固定随机种子, 便于复现结果
# # 对于 ddp 配置, 这个种子**尤其重要**, 因为我们需要确保每个GPU的模型都能初始化成一样的状态!!
# torch.manual_seed(1337)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(1337)
# elif torch.backends.mps.is_available():
#     torch.mps.manual_seed(1337)

# model = GPT(GPTConfig)
# model.to(device)
# # model = torch.compile(model)  # pytorch 2.0 之后支持模型编译, 类似使用 GCC/G++ 之类的编译器编译代码, 而不是直接用python解释器去跑
#                                 # mac上的pytorch目前不支持编译
                                
#                                 # torch.compile() 的功能主要是过一遍整个网络, 然后将一些操作进行整合, 比如 GELU 激活函数, 里面有一些求指数, 开方, 乘加操作, 
#                                 # 由于这些操作都是对同一拨数据按顺序计算, compile的时候将它们直接整合(称为kernel fusion), 这样就可以避免多次数据的存取, 降低延迟, 
#                                 # 因此 torch.compile 是加速模型的一种通用手段
#                                 # 如果是DDP模式,这个 compile 操作会被触发多次, 有n个GPU就会并行地执行n次编译, 这是在CPU上通过ddp开启的线程完成的

# # ddp模式需要用一个container将模型装起来
# if ddp:
#     model = DDP(model, device_ids=[ddp_local_rank])   # 注意, 传进去的时 ddp_local_rank, 即当前机器上的GPU编号
#                                                         # DDP做的事情是: 当每一个GPU都完成了 forward, 用backward计算了模型梯度之后, DDP会执行一个称为 "all reduce" 的操作
#                                                         #             1. 如果当前GPU已经走完 backward, 它就会马上将自己所有参数的梯度经过 NCCL 这个通信后端传给其他GPU, 并开始等待, 等其他GPU传梯度过来
#                                                         #             2. GPU在接收到其他GPU传来梯度时会进行梯度累加, 然后将累加的结果除以GPU总数(我猜是先除GPU数,再累加,这样更快一些), 得到全局平均梯度, 这就完成了 all-reduce 操作

# raw_model = model.module if ddp else model            # 由于model可能已经装到 DDP container 中, 防止后面优化器读取参数除错, 这里需要获取一下没放进container之前的model

# # ------------ optimizer!! ---------------------
# # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)   # adamw 可以当做是 adam 优化器修了一个bug;  
#                                                                 # 3e-4 是大家常用的 early age debug Learning Rate
# # optimizer = model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)  # 这里的 weight_decay 由模型内的一个自定义方法来完成
# optimizer = raw_model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)  # 注意: optimizer 拿参数的时候需要从原模型中获取, 而不是从装进DDP container之后的模型中获取!

# # ------- cosine decay implementation ----------
# max_lr = 6e-4           # GPT-3里的参数
# min_lr = max_lr * 0.1
# warmup_steps = 10
# max_steps = 50          # 最大循环次数, 为了方便演示, 只设置50次

# def get_lr(step):
#     # 1) linear warmup for warmup_iters steps (按照 step 线性增涨)
#     if step < warmup_steps:
#         return max_lr * (step+1) / warmup_steps     # 加1是因为 lr 设置为0没啥意义
    
#     # 2) if step > lr_decay_iters, return min learning rate
#     if step > max_steps:
#         return min_lr
    
#     # 3) in between, use consine decay down to min learning rate
#     decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
#     assert 0 <= decay_ratio <= 1
#     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff start at 1.0 and goes to 0
#     return min_lr + coeff * (max_lr - min_lr)

# # train model
# '''
# 在 GPT-3 paper 的模型超参数表格中, 参数量约为125M的模型对应的 batch size 为 0.5M(这是每次丢进模型的token总数量), lr 为 6e-4
#     因为 batch size 与 lr, AdamW 的 betas 等超参数是存在关联性的, 因此, 如果我们其他地方已经是用了GPT-3的超参数, 那么每次丢进模型的token数量也应该尽可能达到相应的数量.
#         这个方法就是 按照GPU的显存, 设置一个最大的batch size, 但是先不更新参数, 而是将梯度累加, 当 feed 进模型的token数量达到我们想要的数值时才进行参数更新.
# '''
# total_batch_size = 524288 # 0.5M token 这个数不是2的倍数, 这里取 2^19=524288, 某种程度上或许可以让GPU的利用率高些
# B = 4   # micro batch
# T = 1024  # sequence length (注意GPT-2单次最长只支持1024个token)
# assert total_batch_size % (B*T * ddp_world_size) == 0, "Please make sure total_batch_size is divisible by B*T*ddp_world_size!"
# grad_accum_steps = total_batch_size // (B*T * ddp_world_size)   # 计算要走多少个 step 之后才更新参数, 在DDP环境中, 它是将多卡的梯度全部归集到一张卡(累计起来), 然后再统一更新参数的,
#                                                                 # 因此, 要维持原来的 total_batch_size 的话, ddp环境下, 每张卡上跑的step就应该少一些.

# if master_process:    # 我们不希望所有进程都打印一次, 因此只让主线程打印即可
#     print(f"total desired batch size: {total_batch_size}")
#     print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# # ------- 下面这几行命令是用来测试DDP是否正常工作的 ---------
# # print("I am GPU", ddp_rank)
# # print("Bye!")
# # import sys; sys.exit(0)

# # 在shell中, 我们不再是直接用 python + .py 文件的方式运行, 而是在终端中使用 torchrun:
# # torchrun --standalone --nproc_per_node=2 train_gpt2_ning.py
# #       standalone 表示单机; nproc_per_node 表示每个机器上启动多少个进程(通常每个GPU对应一个进程)
# # -----------------------------------------------------


# # 为了确保多卡环境下, 每张GPU加载到不同的数据, 这里的需要改一下我们自己写的 Dataloader
# # train_loader = DataLoaderLite(B=B, T=T)  # batch size 尽可能使用2的倍数, 因为硬件都是2进制, 这样可以让机器运行效率高一些
# train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)                # stage 1~4: 使用 tinyShakespeare dataset

# # torch.set_float32_matmul_precision("high")  # hight: 做乘法的时候使用TF32(精度下降), highest: 做乘法的时候一直使用FP32
#                                                 # 只在 A100之后 的N卡上有用, 在mac上无效

# for step in range(max_steps):
#     t0 = time.time()

#     optimizer.zero_grad()     # 一定以及 清空历史 梯度!!!
#     loss_accum = 0.0
    
#     for micro_step in range(grad_accum_steps):    # 为了使丢进模型的 total batch token 能与 GPT-3 paper 里的超参数匹配, 这里需要使用 accumulate gradient 的技巧(也就是这个内层for循环)
#         x, y = train_loader.next_batch()
#         x, y = x.to(device), y.to(device)
#         # --------- 使用混精度数据类型加速(只有 30系列之后 的 N卡 支持) ------------
#         # with torch.autocast(device_type=device, dtype=torch.bfloat16):  # mac不支持, 只有安培架构(A100之后的显卡)之后才支持
#         #     logits, loss = model(x, y)
#         logits, loss = model(x, y)   # 非混精度模式
#         loss = loss / grad_accum_steps                # 由于这里直接进行了梯度累加, 而计算 loss 的时候, 函数内部是自动求了平均的, 因此这里需要额外除以 grad_accum_steps 进行修正 (本质上是一个乘法分配律) -- 在 当前文件夹下的 play.ipynb 中有代码示例
#         loss_accum += loss.detach()                   # 只是为了打印出来看一看
#         if ddp:
#             model.require_backward_grad_sync = (micro_step == grad_accum_steps -1)    # 因为我们这里在进行梯度累加, 我们并不希望在还没累加完之前就同步梯度, 因此这里用 "require_backward_grad_sync" 控制是否同步
#         loss.backward()    # 计算梯度 (在当前这个嵌套的for循环里, 基于pytorch的特性, 梯度会自动进行累加)
    
#     if ddp:
#         dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)   # 由于 model 丢到了 DDP container中, 因此梯度求了平均; 但 loss的累加并不在 container里, 它不会自动同步所有GPU上的loss
#                                                             # 我们希望打印所有 GPU 上的平均loss, 所以这里手动加一个 all_reduce 操作, 注意求平均可以用 dist.RecudeOp.AVG 来完成

#     norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # GPT-3里提到的操作: 将 global norm of the gradient 给 clip 到 1.0
#                                                                     # 这个函数做的操作: 将所有参数的梯度求平方和, 再开方 (求L2范数); 如果范数的值大于指定的数值, 就会缩放这些梯度来满足指定的范数大小条件; 返回的norm是裁剪之前的梯度范数值.
#                                                                     # 作用: 防止某些batch产生的loss很大, 导致梯度发生剧烈变化, 让训练不稳定
#                                                                     # 通常我们会把这个函数返回的 norm 值打印出来, 如果这个norm一直都挺平稳的, 不会随着训练一直增大, 在一定程度上说明模型正在稳定训练
    
#     # 按照GPT-3文章的描述, lr 按照 consine decay 的方式进行衰减
#     lr = get_lr(step)                                               # 按照 cosine decay 的方式计算 当前的lr
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = lr
    
#     optimizer.step()   # 更新参数
    
#     # torch.cuda.synchronize()   # wait for GPU to finish work (只在 N卡 上有用, mac上无效)
    
#     t1 = time.time()
#     dt = (t1 - t0)  # time difference in seconds
#     tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size   # ddp
#     tokens_per_sec = tokens_processed / dt
#     if master_process:   # 只在 master 进程中打印
#         print(f"step {step} | loss: {loss_accum.item():.6f} | lr: {lr:.5f} | norm: {norm:.4f} | dt: {dt:2f} s | tok/sec: {tokens_per_sec:.2f}")   # loss.item() 可以将tensor换成为 float, 并把数据放回CPU

# if ddp:
#     destroy_process_group()     # 优雅停止ddp进程

# import sys; sys.exit(0)   # 代码走到这里就会停止, 这是一个debug的时候比较不错的方式

# -------------------------------------------------------------------------------------------------------




# ---------------------- stage 5: train GPT-2, DDP 版本, 使用 FineWeb10B 数据集 -----------------------
import time
import tiktoken
import torch.distributed as dist   # DDP 操作的核心模块, 里面包含各种 ddp function
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP   # DDP 模块

# set up DDP (distributed data parallel).
# torchrun sets this env variable
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?  (对于 torchrun 的启动方式来说, 这并不算一种好的检测方式) 
                                            # 这里指定nccl作为后端，nccl是NVIDIA开发的专门用于GPU集群通信的库，能够优化GPU间的通信效率。
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])              # ddp_rank 通常表示当前进程在所有分布式进程中的编号, 如果是单机8卡环境, ddp_rank 数值范围就是 0~7
    ddp_local_rank = int(os.environ['LOCAL_RANK'])  # LOCAL_RANK 表示获取当机器的上局部标识, 也就是当前机器上的GPU ID(rank of the GPU on a single node)
    ddp_world_size = int(os.environ['WORLD_SIZE'])  # world_size 表示所有分布式节点包含的进程数(number of processes), 由于一个GPU对应一个进程, 这个也可以理解为GPU总数
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.   (可以理解为主进程, 表示这个进程需要 兼顾 记录log, 记录checkpoint之类的工作)
else:
    # vanilla, non-DDP
    ddp_rank = 0         # 全局进程ID, 这表示在整个分布式计算环境中, 当前运行的进程ID为0 (因为是单机单卡, 进程就一个, 编号为0)
    ddp_local_rank = 0   # 使用机器上编号为0的GPU
    ddp_world_size = 1   # 单机单卡
    master_process = True # 表示当前进程为主进程

    # attempt to autodetect the device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
    print(f"using device: {device}")

# 固定随机种子, 便于复现结果
# 对于 ddp 配置, 这个种子**尤其重要**, 因为我们需要确保每个GPU的模型都能初始化成一样的状态!!
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
elif torch.backends.mps.is_available():
    torch.mps.manual_seed(1337)

model = GPT(GPTConfig)
model.to(device)
print("Compiling model, please wait....")
model = torch.compile(model)  # pytorch 2.0 之后支持模型编译, 类似使用 GCC/G++ 之类的编译器编译代码, 而不是直接用python解释器去跑
                                # mac上的pytorch目前不支持编译
                                
                                # torch.compile() 的功能主要是过一遍整个网络, 然后将一些操作进行整合, 比如 GELU 激活函数, 里面有一些求指数, 开方, 乘加操作, 
                                # 由于这些操作都是对同一拨数据按顺序计算, compile的时候将它们直接整合(称为kernel fusion), 这样就可以避免多次数据的存取, 降低延迟, 
                                # 因此 torch.compile 是加速模型的一种通用手段
                                # 如果是DDP模式,这个 compile 操作会被触发多次, 有n个GPU就会并行地执行n次编译, 这是在CPU上通过ddp开启的线程完成的

# ddp模式需要用一个container将模型装起来
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])   # 注意, 传进去的时 ddp_local_rank, 即当前机器上的GPU编号
                                                        # DDP做的事情是: 当每一个GPU都完成了 forward, 用backward计算了模型梯度之后, DDP会执行一个称为 "all reduce" 的操作
                                                        #             1. 如果当前GPU已经走完 backward, 它就会马上将自己所有参数的梯度经过 NCCL 这个通信后端传给其他GPU, 并开始等待, 等其他GPU传梯度过来
                                                        #             2. GPU在接收到其他GPU传来梯度时会进行梯度累加, 然后将累加的结果除以GPU总数(我猜是先除GPU数,再累加,这样更快一些), 得到全局平均梯度, 这就完成了 all-reduce 操作

raw_model = model.module if ddp else model            # 由于model可能已经装到 DDP container 中, 防止后面优化器读取参数除错, 这里需要获取一下没放进container之前的model

# ------------ optimizer!! ---------------------
optimizer = raw_model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)  # 注意: optimizer 拿参数的时候需要从原模型中获取, 而不是从装进DDP container之后的模型中获取!

# ------- cosine decay implementation ----------
max_lr = 6e-4           # GPT-3里的参数
min_lr = max_lr * 0.1
# stage 5: 使用 FineWeb10B dataset
warmup_steps = 715        # 数值由来: GPT-3 paper 里提到 warmup 一共用了 375M token, batch size token数量为 0.5M, 我们取最接近的2的倍数, 即2^19; 于是 375e6 / 2**19 ≈ 715 steps
max_steps = 19073         # 数值由来: 数据集一共10B tokens, 也就是 10e9, 按照GPT-3里的 batch size token数量(0.5M), 我们取最接近的2的倍数, 即2^19; 而 10e9 / (2**19) ≈ 19073 steps

def get_lr(step):
    # 1) linear warmup for warmup_iters steps (按照 step 线性增涨)
    if step < warmup_steps:
        return max_lr * (step+1) / warmup_steps     # 加1是因为 lr 设置为0没啥意义
    
    # 2) if step > lr_decay_iters, return min learning rate
    if step > max_steps:
        return min_lr
    
    # 3) in between, use consine decay down to min learning rate
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff start at 1.0 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# train model
'''
在 GPT-3 paper 的模型超参数表格中, 参数量约为125M的模型对应的 batch size 为 0.5M(这是每次丢进模型的token总数量), lr 为 6e-4
    因为 batch size 与 lr, AdamW 的 betas 等超参数是存在关联性的, 因此, 如果我们其他地方已经是用了GPT-3的超参数, 那么每次丢进模型的token数量也应该尽可能达到相应的数量.
        这个方法就是 按照GPU的显存, 设置一个最大的batch size, 但是先不更新参数, 而是将梯度累加, 当 feed 进模型的token数量达到我们想要的数值时才进行参数更新.
'''
total_batch_size = 524288 # 0.5M token 这个数不是2的倍数, 这里取 2^19=524288, 某种程度上或许可以让GPU的利用率高些
B = 4   # micro batch
T = 1024  # sequence length (注意GPT-2单次最长只支持1024个token)
assert total_batch_size % (B*T * ddp_world_size) == 0, "Please make sure total_batch_size is divisible by B*T*ddp_world_size!"
grad_accum_steps = total_batch_size // (B*T * ddp_world_size)   # 计算要走多少个 step 之后才更新参数, 在DDP环境中, 它是将多卡的梯度全部归集到一张卡(累计起来), 然后再统一更新参数的,
                                                                # 因此, 要维持原来的 total_batch_size 的话, ddp环境下, 每张卡上跑的step就应该少一些.

if master_process:    # 我们不希望所有进程都打印一次, 因此只让主线程打印即可
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# 为了确保多卡环境下, 每张GPU加载到不同的数据, 这里的需要改一下我们自己写的 Dataloader, 为了支持 train/val split, 这里还改了读取数据的方式
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")   # stage 5: 使用 FineWeb10B dataset
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")       # stage 5: 使用 FineWeb10B dataset

# 下面这个功能只在 A100之后 的N卡上有用, 在mac上无效
torch.set_float32_matmul_precision("high")  # hight: 做乘法的时候使用TF32(精度下降), highest: 做乘法的时候一直使用FP32

for step in range(max_steps):
    t0 = time.time()

    # validation loop: once in a while evaluate our validation loss
    if step % 100 == 0:
        model.eval()
        val_loader.reset()              # 初始化数据读取状态: 读取第一个 val set 的 shard 0
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20         # evaluation 跑20个step
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                # # 混精度推理 (A100之后的GPU才支持, 3090 不支持)
                # with torch.autocast(device_type=device, dtype=torch.bfloat16):
                #     logits, loss = model(x, y)
                logits, loss = model(x, y)
                loss /= val_loss_steps
                val_loss_accum += loss.detach()   # 一定不要忘记 detach !!!!
        if ddp:
            dist.all_reduce(val_loss_accum, dist.ReduceOp.AVG)    # 将多卡的累计loss数值同步, 并求平均
        if master_process:
            print(f"validation loss: {val_loss_accum.item():4f}") # 限定只有主线程打印log

    # train loop
    model.train()
    model.zero_grad()
    loss_accum = 0.0
    # 内层循环: 梯度累加, 以达到GPT-3文章中的batch token数量
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # # 混精度推理 (A100之后的GPU才支持, 3090 不支持)
        # with torch.autocast(device_type=device, dtype=torch.bfloat16):
        #     logits, loss = model(x, y)
        logits, loss = model(x, y)
        loss /= grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps -1)  # 标记当前循环是否在 backward 时触发 all_reduce
        loss.backward()           # 梯度累加 & 检查标志自动触发 all_reduce
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 梯度裁剪, 让模型训练更稳定

    lr = get_lr(step)
    for param_group in optimizer.param_groups:   # 为每一组参数(这里的分组是指 "参数有无weight decay") 分别设置 Learning rate
        param_group["lr"] = lr

    optimizer.step()                # 到这里, 梯度已经是全局平均梯度, lr 也已经设置完成, 开始更新权重参数 
    torch.cuda.synchronize()        # DDP环境下, 等待所有GPU完成参数更新再进行后续操作
    t1 = time.time()
    dt = t1 - t0                    # 单位毫秒
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:              # 只在 master 进程中打印
        print(f"step {step} | loss: {loss_accum.item():.6f} | lr: {lr:.5f} | norm: {norm:.4f} | dt: {dt:2f} s | tok/sec: {tokens_per_sec:.2f}") 

if ddp:
    destroy_process_group()

import sys; sys.exit(0)   # 代码走到这里就会停止, 这是一个debug的时候比较不错的方式


