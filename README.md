## 简述
CharacterPortrait 是一个基于大语言模型的智能人物画像生成及运用系统。该系统利用 ChatGLM4 大模型，结合自然语言处理和知识图谱技术，通过分析与用户的对话内容，自动构建全方位的用户画像，并在用户画像的基础上，进行后续的对话内容生成。系统可以从性格特征、兴趣爱好、价值观念、职业背景等多个维度对用户进行深入分析，为个性化服务和用户洞察提供数据支持。✨

主要特点：
- 支持实时对话分析和用户画像更新
- 多维度用户特征提取与分析
- 基于知识图谱的用户画像构建
- 支持批量用户数据处理
- 提供友好的 Web 交互界面

### 预训练BERT和tokenizer配置
在 character_portrait.py 文件中需加载预训练的 BERT 和 tokenizer。因模型较大将导致clone耗时较长, 本仓库未加入这部分内容, 初次使用时需要科学上网从远程服务器加载下来（也可自行在model文件夹下添加text2vec-base-chinese文件）。

### API密钥配置
在 `config/api_keys.py` 文件中配置智谱AI的API密钥：
```python
ZHIPUAI_API_KEYS = {
    "chat_key": "your_chat_api_key",  # 用于对话系统
    "profile_keys": [  # 用于用户画像更新
        "your_profile_key_1",
        "your_profile_key_2",
        "your_profile_key_3",
        "your_profile_key_4"
    ]
}
```
- `chat_key`: 用于主对话系统的API密钥
- `profile_keys`: 用于用户画像更新的多个API密钥，支持并行处理

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行程序
```bash
python character_portrait.py
```