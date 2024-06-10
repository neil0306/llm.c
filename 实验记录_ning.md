# 搭建环境
根据 https://github.com/karpathy/llm.c/discussions/481 的叙述进行实验, 搭环境的区别:
1. 直接使用了 conda 而不是 miniconda 构建虚拟环境
2. 为了节省系统盘空间, llm.c 仓库丢在 980_1T 硬盘下面, 创建了 softlink 到 Desktop
   1. 对应地, `dev/data/fineweb10B` 这里下载的数据也会放在 980_1T 硬盘下
   2. `~46GB in ~/.cache/huggingface/datasets/HuggingFaceFW___fineweb` 这46G的数据应该还会存放在系统盘里
3. pytorch没有使用 fortnightly 版本, 而是使用了 stable 版本
4. 由于机器上有 2080Ti, 也有3090, 执行`make train_gpt2cu USE_CUDNN=1`, 编译时获取架构 `$(GPU_COMPUTE_CAPABILITY)` 会返回两个版本: 75 和 86
   1. 75 对应 2080Ti
   2. 86 对应 3090

修改好GPU架构后, 执行`make train_gpt2cu USE_CUDNN=1`时, 仍会报错:
```shell
fatal error: nccl.h: No such file or directory
```
网上查解决方案, 作如下尝试:
```shell
git clone https://github.com/NVIDIA/nccl.git
cd nccl
sudo make install -j4    # 编译比较久
sudo ldconfig
```
- 博客链接: https://arc.net/l/quote/hrcampsj
- [x] update: 安装完后编译成功, 已生成 `train_gpt2cu`

# 跑训练

```shell
mpirun -np 2 ./train_gpt2cu \
    -i "dev/data/fineweb10B/fineweb_train_*.bin" \
    -j "dev/data/fineweb10B/fineweb_val_*.bin" \
    -o ./log124M \
    -e "d12" \
    -b 32 -t 1024 \
    -d 524288 \
    -r 1 \
    -z 1 \
    -c 0.1 \
    -l 0.0006 \
    -q 0.0 \
    -u 700 \
    -n 5000 \
    -v 250 -s 20000 \
    -h 1
```
- 使用上面编译好的 `train_gpt2cu` 进行训练, 训练命令如上
- batch size 为 `32`, 原参数是64, 会 out-of-memory
- 训练的时候需要手动创建记录日志的目录, 这里使用原作者给的名字`log124M`

参数含义:

- `-i` `-j` are training and validation splits token files, written by fineweb.py

- `-o` is the output directory to write logs and checkpoints into

- `-e` "d12" asks to initialize, a depth 12 GPT-2 model from scratch

- `-b` 64 sets the micro-batch size to 64 . If you are running out of memory, decrease this value, e.g. try 32, 16, 8, all the way down to 1 potentially.

- `-t` 1024 sets the maximum sequence length to 1024, as GPT-2 did

- `-d` 524288 requests that the total batch size per single update be ~0.5M tokens. The code will take this desired batch size and calculate the needed gradient accumulation "inner loop" steps of the optimization. 
  - For example on 8 GPUs, at -b 64 and -t 1024, every microbatch is doing exactly 8 X 64 X 1024 = 524288 tokens, so there is no need for gradient accumulation. 
  - But if we we only have 1 GPU, then the code will set it to 8, and do an inner loop of 8 iterations to add up to this "total batch size" per step. While the batch size used to train GPT-2 is unknown, this number ~0.5M comes from the GPT-3 paper table, for this model size.

- `-r `1 sets the recompute setting = 1, so we will re-compute the GeLU activations. This slightly increases the runtime, but saves quite a bit of memory, allowing us to increase the batch size and get a net increase in token throughput.

- `-z` 1 turns on ZeRO-1 (i.e. optimizer state sharding) across multiple GPUs. If you're training with > 1 GPU, this setting is a no-brainer and should basically always be on. On 1 GPU this setting is a no-op.

- `-c` 0.1 sets the weight decay to 0.1. Only (2D) weights are decayed exactly as in GPT-2, and this number comes from the GPT-3 paper

- `-l` 0.0006 sets the maximum learning rate, from GPT-3 paper.

- `-q` 0.0 says that we will decay the learning rate to 0 over the course of training.

- `-u` 700 says that we will ramp up the learning rate from 0 to max learning rate over the first 700 iterations, which at total batch size 0.5M is 350M tokens, following GPT-3 paper.

- `-n` 5000 asks to save model checkpoints every 5000 steps.

- `-v` 250 asks to evaluate and log the validation loss every 250 steps

- `-s` 20000 asks to sample some tokens every 20000 steps. Because the total number of steps will be less than this (see below), this basically turns generation off and we will only basically sample a single time at the very end.

- `-h` 1 asks to evaluate the HellaSwag accuracy, something we can compare across papers.

- Because we did not set the maximum number of steps using -x flag, it defaults to exactly one epoch over the training data, i.e. 10B tokens. Because the total batch size is ~0.5M and total number of tokens is 10B, there will be a total of ~ 10B/0.5M = 20K steps.

# tensorboard 查看训练log
由于代码中已经将log输出到`log124M`文件夹下的 `main.log` 中, 所以这里需要的步骤是:
1. 安装tensorboard
   ```shell
   conda activate 跑训练使用的虚拟环境名称
   pip install tensorboard   

   # 由于网络问题, 如果下载很慢, 可以临时使用清华源
   pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```
2. 进入`log124M`文件夹, 执行`tensorboard --logdir .`
   - 或者用vscode, 安装`Tensorboard`插件, 用快捷键`command+shift+p`打开命令面板, 输入`Tensorboard: Launch Tensorboard`, 选择`log124M`文件夹, 点击`Launch`即可


