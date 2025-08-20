## 目录说明
### demo01_firstBuildLlm
从零开始200行python代码实现LLM
#### 介绍
尝试从零开始，用python实现一个极简但完整的大语言模型，在过程中把各种概念“具象化”，写出self-attention机制、transformer模型，亲自感受下训练、推理中会遇到的一些问题
#### 文件说明
**simplemodel.py**: 传统方式实现一个“诗词生成器”。通过计算每个字后面出现各个字的概率，然后根据这些概率，不断的递归生成“下一个字”，截断一部分，就是一首词了

**simplebigrammodel.py** 把Bigram Language Model的实现变得更加“机器学习风格”,引入了类结构、批处理和并行生成，适用于多个提示词