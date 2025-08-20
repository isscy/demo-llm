import random
from typing import List

# 把Bigram Language Model的实现变得更加“机器学习风格”,引入了类结构、批处理和并行生成，适用于多个提示词
# 核心是二元语言模型，基于马尔可夫链原理，通过统计训练文本（ci.txt）中相邻字符的转移频率，构建转移概率矩阵

#  step-0: 准备工作
random.seed(42) # 去掉此行，获得随机结果
prompts = ["春江", "往事"] # 包含多个起始提示词
max_new_token = 100 # 每个提示词生成的最大新字符数
max_iters = 8000 # 训练迭代次数，控制训练数据采样的次数
batch_size = 32 # 每次训练的批次大小，即并行处理的序列数
block_size = 8 # 每个序列的长度，用于训练和生成
with open('demo01_firstBuildLlm/data/ci.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#  step-1: 定义 Tokenizer 类
class Tokenizer:
    def __init__(self, text: str):
        self.chars = sorted(list(set(text))) # 唯一字符的排序列表
        self.vocab_size = len(self.chars) # 字符集大小
        self.stoi = {ch: i for i, ch in enumerate(self.chars)} # 字符到索引的字典（如 {'春': 0, '江': 1}）
        self.itos = {i: ch for i, ch in enumerate(self.chars)} # 索引到字符的字典（如 {0: '春', 1: '江'}）

    def encode(self, s: str) -> List[int]: # 将字符串 s 转换为整数列表
        return [self.stoi[c] for c in s]

    def decode(self, l: List[int]) -> str: # 将整数列表 l 转换回字符串
        return ''.join([self.itos[i] for i in l])

# 创建转移矩阵 transition，transition[i][j] 表示从字符 i 转移到字符 j 的频率
class BigramLanguageModel():
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        # transition[i][j] 表示从字符 i 转移到字符 j 的频率
        self.transition = [[0 for _ in range(vocab_size)]
                           for _ in range(vocab_size)]
    # __call__ 是 Python 的特殊方法，允许对象像函数一样被调用
    # model(x)，等价于 model.forward(x)
    def __call__(self, x):
        # 方便直接调用model(x)
        return self.forward(x)

    def forward(self, idx: List[List[int]]) -> List[List[List[float]]]:
        '''
        输入idx，是一个二维数组，如[[1, 2, 3],
                                  [4, 5, 6]]
        表示同时希望推理的多个序列

        输出是一个三维数组，如[[[0.1, 0.2, 0.3, .. (vocab_size)],
                                [0.4, 0.5, 0.6, .. (vocab_size)],
                                [0.7, 0.8, 0.9, .. (vocab_size)]],

                               [[0.2, 0.3, 0.4, .. (vocab_size)],
                                [0.5, 0.6, 0.7, .. (vocab_size)],
                                [0.8, 0.9, 1.0, .. (vocab_size)]]]

        '''
        B = len(idx)  # 批次大小
        T = len(idx[0])  # 每一批的序列长度

        logits = [
            [[0.0 for _ in range(self.vocab_size)]
             for _ in range(T)]
            for _ in range(B)
        ]

        for b in range(B):
            for t in range(T):
                current_token = idx[b][t]
                # 计算了每一个token的下一个token的概率
                logits[b][t] = self.transition[current_token]

        return logits

    def generate(self, idx: List[List[int]], max_new_tokens: int) -> List[int]:
        for _ in range(max_new_tokens):
            logits_batch = self(idx)
            for batch_idx, logits in enumerate(logits_batch):
                # 我们计算了每一个token的下一个token的概率
                # 但实际上我们只需要最后一个token的“下一个token的概率”
                logits = logits[-1]
                total = max(sum(logits), 1)
                # 归一化
                logits = [logit / total for logit in logits]
                # 根据概率随机采样
                next_token = random.choices(
                    range(self.vocab_size),
                    weights=logits,
                    k=1
                )[0]
                idx[batch_idx].append(next_token)
        return idx


def get_batch(tokens, batch_size, block_size):
    '''
    随机获取一批数据x和y用于训练
    x和y都是二维数组，可以用于并行训练
    其中y数组内的每一个值，都是x数组内对应位置的值的下一个值
    格式如下：
    x = [[1, 2, 3],
         [9, 10, 11]]
    y = [[2, 3, 4],
         [10, 11, 12]]
    '''
    ix = random.choices(range(len(tokens) - block_size), k=batch_size)
    x, y = [], []
    for i in ix:
        x.append(tokens[i:i + block_size])
        y.append(tokens[i + 1:i + block_size + 1])
    return x, y

# 初始化和训练模型
tokenizer = Tokenizer(text)
vocab_size = tokenizer.vocab_size

tokens = tokenizer.encode(text)

model = BigramLanguageModel(vocab_size)

# 训练
for iter in range(max_iters):
    x_batch, y_batch = get_batch(tokens, batch_size, block_size)
    for i in range(len(x_batch)):
        for j in range(len(x_batch[i])):
            x = x_batch[i][j]
            y = y_batch[i][j]
            model.transition[x][y] += 1

prompt_tokens = [tokenizer.encode(prompt) for prompt in prompts]

# 推理
result = model.generate(prompt_tokens, max_new_token)

# decode
for tokens in result:
    print(tokenizer.decode(tokens))
    print('-' * 10)