# 关于GPU参数
![](notes_about_dtype_images/A100参数.png)
- A100GPU支持的最高精的度浮点数类型是 **float64 (FP64)**, 此时它的算力是 `9.7 TFLOPS`, 这表示它一秒内能做 9.7 trillion 次 64位浮点数的浮点运算
  - 如果是转换成 Tensor Core 专用的数据类型的话(**同样是64位浮点数**, 应该是表示数值的方法不一样, 遵循的标准不一样), 它的算力是 `19.5 TFLOPS`, 这表示它一秒内能做 19.5 trillion 次 64位浮点数的浮点运算
- 如果精度降低一些, 使用 **float32 (FP32)**, 它的算力是 `19.5 TFLOPS`, 这表示它一秒内能做 19.5 trillion 次 32位浮点数的浮点运算
  - 使用Tensor Core专用的数据类型的话, 它的算力是 `156 TFLOPS`, 这表示它一秒内能做 156 trillion 次 32位浮点数的浮点运算 (右侧带星号的数值312指的是 稀疏表示 的情况, 大部分情况下如果不进行针对性优化是不关注这个数值的)
- 从32位精度下降到 16 位(半精度)的话, 就是 `BFLOAT16 Tensor Core`类型, 对应的算力是 `312 TFLOPS`, 这表示它一秒内能做 312 trillion 次 16位浮点数的浮点运算
  - 同样, BFP16 也有对应的 Tensor Core 类型, 不过此时的算力是相同的.
- 对于 INT8 类型, **它只能用在 inference 阶段, 不能用于模型训练**, 它的算力是 `624 TFLOPS`, 这表示它一秒内能做 624 trillion 次 8位整数的整数运算.

在训练模型的时候, 如果我们从 nvidia-smi 命令中看到 GPU 的util百分比不高的话, 说明GPU大部分时间是在等待数据的加载, 而不是在进行浮点数计算.

GPU的FLOPS通常指的是`乘,加运算`, 也就是做矩阵乘法的时候的一次加法和一次乘法.
- GPU在做矩阵乘法时, 为了使效率更高, 通常会将大矩阵拆分成4x4的小矩阵进行相乘, 然后再拼回一个大矩阵, 如下:
    ![](notes_about_dtype_images/GPU中的4x4矩阵乘法示意图.png)

## 关于TF32与FP32在算力上的区别
- 从下图的上半部分可以看到, TF32 和 FP32 只是在精度位数上存在区别, TF32的精度位数从23位截断到10位.  
    ![](notes_about_dtype_images/A100在TF32和FP32在计算时使用的精度截断.png)
  - A100 GPU 在矩阵乘法运算中, 虽然 Input 和 Output 使用的都是 FP32 的数据表示方法, 但是在进行`乘法`的时候将精度截断了10位, 使得精度降低但速度变快, 然后到累加器中, 使用仍然是FP32, 维持乘法输出的精度.
  - 图片来自[A100GPU架构白皮书](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)

在使用 A100 GPU时, 我们只需要在代码中设置`torch.set_float32_matmul_precision("high")`, 就能让矩阵乘法加速, 不过同时也会带来精度上的损失.
- 注意: 这只是在矩阵乘法的中间过程中短暂地使用了TF32类型, 在整个网络的任意位置查看参数`dtype`的时候仍然是 FP32 类型.
```python
# train model
import time
train_loader = DataLoaderLite(B=4, T=32)  # batch size 尽可能使用2的倍数, 因为硬件都是2进制, 这样可以让机器运行效率高一些

torch.set_float32_matmul_precision("high")  # hight: 做乘法的时候使用TF32(精度下降), highest: 做乘法的时候仍然使用FP32
                                                # 只在 A100之后 的N卡上有用, 在mac上无效

for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()     # 一定以及 清空历史 梯度!!!
    logits, loss = model(x, y)
    
    # import code; code.interact(local=locals())   # 通过这行代码, 我们可以在终端触发一个 interactive console, 直接进行一些debug操作
    
    loss.backward()    # 计算梯度
    optimizer.step()   # 更新参数
    
    # torch.cuda.synchronize()   # wait for GPU to finish work (只有在N卡上有用, mac上无效)
    
    t1 = time.time()
    dt = (t1 - t0) * 1000 # time difference in miliseconds
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step {i}, loss: {loss.item()}, dt: {dt:2f}ms, tok/sec: {tokens_per_sec}")   # loss.item() 可以将tensor换成为 float, 并把数据放回CPU

import sys; sys.exit(0)   # 代码走到这里就会停止, 这是一个debug的时候比较不错的方式
```
