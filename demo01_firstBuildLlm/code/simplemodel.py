import random

random.seed(42)  # 固定随机种子 去掉此行，获得随机结果

prompt = "春江" # 指定生成文本的起始字符串，作为生成序列的开头
max_new_token = 100 # 限制生成的最大字符数（不包括提示词的字符）

with open('demo01_firstBuildLlm/data/ci.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# chars: 包含文本中所有唯一字符的排序列表
chars = sorted(list(set(text))) # set:将文本字符串转换为去重集合 list:集合转为列表 sorted:对字符列表按 Unicode 排序
vocab_size = len(chars) #  vocab_size：字符集的大小，即词汇表中不同字符的数量
#stoi：字典推导式，键是字符，值是对应的索引。例如，{'春': 0, '江': 1, ...}。
stoi = { ch:i for i,ch in enumerate(chars) }
#itos：字典推导式，键是索引，值是对应的字符。例如，{0: '春', 1: '江', ...}。
itos = { i:ch for i,ch in enumerate(chars) }
#encode：将字符串转为整数序列。例如，encode("春江") 可能返回 [0, 1]。
encode = lambda s: [stoi[c] for c in s]
#decode：将整数序列转为字符串。例如，decode([0, 1]) 返回 "春江"。
decode = lambda l: ''.join([itos[i] for i in l])

#构建转移概率矩阵
transition = [[0 for _ in range(vocab_size)] for _ in range(vocab_size)]

for i in range(len(text) - 1):
    current_token_id = encode(text[i])[0]
    next_token_id = encode(text[i + 1])[0]
    transition[current_token_id][next_token_id] += 1

generated_token = encode(prompt)

for i in range(max_new_token - 1):
    current_token_id = generated_token[-1]
    logits = transition[current_token_id]
    total = sum(logits)
    logits = [logit / total for logit in logits]
    next_token_id = random.choices(range(vocab_size), weights=logits, k=1)[0]
    generated_token.append(next_token_id)
    current_token_id = next_token_id

print(decode(generated_token))