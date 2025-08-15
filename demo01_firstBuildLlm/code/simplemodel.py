import random
# 传统方式实现一个“诗词生成器”
### 通过计算每个字后面出现各个字的概率，然后根据这些概率，不断的递归生成“下一个字”，截断一部分，就是一首词了

#  step-0: 准备工作
random.seed(42)  # 固定随机种子 去掉此行，获得随机结果

prompt = "春江" # 指定生成文本的起始字符串，作为生成序列的开头
max_new_token = 100 # 限制生成的最大字符数（不包括提示词的字符）

with open('demo01_firstBuildLlm/data/ci.txt', 'r', encoding='utf-8') as f:
    text = f.read()
# step-1: 构建字符词汇表
### chars: 包含文本中所有唯一字符的排序列表
chars = sorted(list(set(text))) # set:将文本字符串转换为去重集合 list:集合转为列表 sorted:对字符列表按 Unicode 排序
vocab_size = len(chars) #  vocab_size：字符集的大小，即词汇表中不同字符的数量
# step-2: 创建字符编码和解码映射
### stoi：字典推导式，键是字符，值是对应的索引。例如，{'春': 0, '江': 1, ...}。
stoi = { ch:i for i,ch in enumerate(chars) }
### itos：字典推导式，键是索引，值是对应的字符。例如，{0: '春', 1: '江', ...}。
itos = { i:ch for i,ch in enumerate(chars) }
### encode：将字符串转为整数序列。例如，encode("春江") 可能返回 [0, 1]。
encode = lambda s: [stoi[c] for c in s]
### decode：将整数序列转为字符串。例如，decode([0, 1]) 返回 "春江"。
decode = lambda l: ''.join([itos[i] for i in l])

# step-3: 构建转移概率矩阵 : 列表推导式创建二维列表，尺寸为 vocab_size × vocab_size, 每个元素初始化为 0
###  transition 是一个矩阵，transition[i][j] 表示从字符 i 转移到字符 j 的出现次数
###  例如，如果 text 中 "春" 后面经常出现 "江"，则 transition[stoi['春']][stoi['江']] 的值会较大
transition = [[0 for _ in range(vocab_size)] for _ in range(vocab_size)]
### --------------------------------------------------------------BEGIN
#      a    b    c  ... (vocab_size)
#   a  0    1    0  ...
#   b  0    0    3  ...
#   c  4    0    0  ...
#  ... ... ... ... ... ... ...
# (vocab_size)
# vocab_size * vocab_size的二维数组，记录每个词的下一个词的出现次数
### --------------------------------------------------------------END

# step-4: 统计字符转移频率
### 遍历文本，处理每对相邻字符（到倒数第二个字符）
### 遍历 text 中的字符对，例如 "春江"，将 "春" 编码为 current_token_id，"江" 编码为 next_token_id
for i in range(len(text) - 1):
    ### 将第 i 个字符编码为整数。由于 encode 返回列表，取 [0] 获取单个整数
    current_token_id = encode(text[i])[0]
    next_token_id = encode(text[i + 1])[0]
    ### 在转移矩阵中记录相邻字符的出现次数
    transition[current_token_id][next_token_id] += 1

# step-5: 初始化生成序列
### 将提示词 "春江" 转换为整数列表
### generated_token 是一个整数列表，初始化为提示词的编码。例如，[stoi['春'], stoi['江']]，作为生成序列的起点
generated_token = encode(prompt)

# step-6: 生成新字符: 基于马尔可夫链模型，从当前字符的转移概率分布中随机选择下一个字符
### 一阶马尔可夫链，假设下一个字符只依赖于当前字符（而非更长的历史）
### 循环生成最多 max_new_token - 1 个新字符（减 1 是因为提示词已占一部分）
for i in range(max_new_token - 1):
    ### 取当前序列的最后一个字符的编码
    current_token_id = generated_token[-1]
    ### 获取当前字符转移到所有可能字符的频率列表
    ### 例如，如果当前字符是 "江"，logits 是 transition[stoi['江']]，表示 "江" 后面可能出现的字符的频率
    logits = transition[current_token_id]
    ### 计算转移频率的总和
    total = sum(logits)
    ### 将频率归一化为概率（每个值除以总和）
    ### --------------------------------------------------------------BEGIN
    # 归一化前 logits = [0, 10, ...., 298,..., 88, ..., 13, 0]
    #         len(logits) = vocab_size, sum(logits) = 6664
    # 归一化后 logits = [0/6664, 10/6664, ...., 298/6664,..., 88/6664, ..., 13/6664, 0/6664]
    #                = [0, 0.0015, ..., 0.0447, ..., 0.0132, ..., 0.00195, 0]
    ### --------------------------------------------------------------END
    logits = [logit / total for logit in logits]
    ### 根据概率随机选择下一个字符的索引
    ### 归一化后，logits 变为概率分布，random.choices 根据这些概率随机选择下一个字符
    next_token_id = random.choices(range(vocab_size), weights=logits, k=1)[0]
    ### 将新字符的索引添加到序列中
    ### 新字符的索引添加到 generated_token 中，循环继续，直到生成指定数量的字符
    generated_token.append(next_token_id)
    ### 更新当前字符为新生成的字符
    current_token_id = next_token_id

print(decode(generated_token))