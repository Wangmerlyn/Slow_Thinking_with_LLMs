#!/bin/bash

# 文件路径
file_path="/mnt/longcontext/models/siyuan/rl_datasets/longcontext_train_30k/train.jsonl"

# 要查找的句子
target_sentence="Where was the director of film Fury Of The Pagans born?"

# 输出文件
output_file="output.txt"

# 使用 grep 查找句子，-i 选项忽略大小写，-n 选项显示行号，-F 选项表示精确匹配
# 将匹配的行输出到 output.txt 文件
grep -inF "$target_sentence" "$file_path" >> "$output_file"

# 检查是否找到了匹配的句子
if [ $? -eq 0 ]; then
    echo "Sentence found and written to $output_file."
else
    echo "Sentence not found."
fi