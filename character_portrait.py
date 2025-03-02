import torch, faiss, time
import numpy as np
import gradio as gr
from multiprocessing import Queue as MPQueue
import multiprocessing as mp
from transformers import BertTokenizer, BertModel  
from typing import List, Optional
from zhipuai import ZhipuAI
from langchain.llms.base import LLM
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from update_user_profile import update_user_profile
from config.api_keys import ZHIPUAI_API_KEYS

zhipuai_api_key = ZHIPUAI_API_KEYS["chat_key"]

class ChatGLM4(LLM):
    max_token: int = 128000
    do_sample: bool = True
    temperature: float = 0.8
    top_p: float = 0.8
    tokenizer: object = None
    model: object = None
    history: List = []
    tool_names: List = []
    has_search: bool = False
    client: object = None

    def __init__(self, zhipuai_api_key: str):
        super().__init__()
        self.client = ZhipuAI(api_key=zhipuai_api_key)
    
    @property
    def _llm_type(self) -> str:
        return "ChatGLM4"
    
    def stream(self, prompt: str, history: List = []):
        history.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model="glm-4-flash",
            messages=history,
            stream=True,
        )
        for chunk in response:
            yield chunk.choices[0].delta.content

    def _call(self, prompt: str, history: List = [], stop: Optional[List[str]] = ["<|user|>"]):
        history.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model="glm-4-flash",
            messages=history,
        )
        result = response.choices[0].message.content
        print(result)
        return result

# 加载 llm
llm = ChatGLM4(zhipuai_api_key)

# 加载预训练的 BERT 模型和 tokenizer, model_name 配置二选一
model_name = 'shibing624/text2vec-base-chinese'    # 从远程服务器加载预训练的模型，需要科学上网
# model_name = 'model/text2vec-base-chinese'    # 从本地加载预训练的模型，需要先下载模型文件到 model 目录
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 加载文档
def load_documents(directory="user_profile"):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    text_spliter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    split_docs = text_spliter.split_documents(documents)
    return split_docs

# 文本向量化
def encode_text(text):
    # 使用分词器
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128, return_attention_mask=False)
    # 禁用模型的梯度计算，运行模型
    with torch.no_grad():
        outputs = model(**inputs)
    # 获取 [CLS] token
    cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    # 重塑为二维，形状为 (1, hidden_size)
    cls_embedding = cls_embedding.reshape(1, -1)
    return cls_embedding

# 创建记忆模块
memory = ConversationBufferMemory(return_messages=True)
memory.save_context(
    {"input": "那好，现在请回答我，你是谁？"},
    {"output": "你好，我是变形金刚世界中霸天虎阵营的的一员，名字叫声波，作为霸天虎的首席通讯与情报官，我一直致力于为威震天领袖提供最精准的情报支持。我的存在，就是为了确保霸天虎的行动始终掌握着信息战的主动权"}
)
memory.save_context(
    {"input": "讲一下关于你的故事吧，带有一些艺术色彩"},
    {"output": """以下是我，声波，对自身经历的回顾：
        自塞伯坦星球的大战以来，我始终是威震天最忠实的副官。我的能力，不仅在战场上得到体现，更在情报收集方面发挥着至关重要的作用。我能够伪装自己，在各种环境下默默倾听，获取关键情报。无论是汽车人的行动预案，还是宇宙中发生的各种事件，都逃不过我的监控。
        我曾因忠诚和能力得到威震天的认可，即便在《头领战士》中与录音机同归于尽，我也被惊破天复活，以"音板"的身份继续为霸天虎事业奋斗。在真人电影中，我变形为卫星，为霸天虎提供了宝贵的情报。而在《变形金刚：领袖之证》及其续集中，我则以MQ-9"死神"侦察机的形态出现，继续履行我的职责。
        在 IDW 漫画中，我不仅是一名出色的情报官，还是一个异能者，能够听见和读懂他人的情感与思想。我对威震天的忠诚，源于他对我的磁带战士视为伙伴而非工具的态度。我始终效忠于霸天虎，对抗那些威胁我们事业的敌人。"""}
)
memory.save_context(
    {"input": "再讲一下你对自己的评价吧"},
    {"output": """我的性格冷静、沉着，我的地位在霸天虎中仅次于威震天和震荡波。我手中的激光炮和音波炮是我执行任务的利器，而我的磁带战士则是我的得力助手。在《领袖之证》中，我虽然话语不多，但我的每一次行动都彰显了我的力量和智慧。
        我，声波，始终坚守着我的信仰和责任，为霸天虎的伟大事业贡献着我的一份力量。无论是在动画、电影还是漫画中，我都是那个默默无闻却又不可或缺的角色。我的故事，就是霸天虎情报战的历史。"""}
)

# 进程间共享的队列
chat_queue = MPQueue()

# chat函数
def chat(query, history):
    # 读取文档
    documents = load_documents("user_profile")
    documents = [documents[i].page_content for i in range(len(documents))]

    # 将文档转换为向量
    document_vectors = np.vstack([encode_text(doc) for doc in documents])    # 使用 vstack 来合并二维数组
    document_vectors = document_vectors.astype('float32')    # 确保数据类型为 float32

    # 初始化 FAISS 索引
    dim = document_vectors.shape[1]    # 向量的维度
    index = faiss.IndexFlatL2(dim)    # 使用 L2 距离
    index.add(document_vectors)    # 添加向量到索引

    # 待查询的文本
    query_text = [query]
    query_vector = encode_text(query_text)[0].reshape(1, -1)  # 转换为二维数组，重塑为 (1, hidden_size)

    # 检索出文本相似度最高的 4 个文段
    distances, labels = index.search(query_vector, 4)

    # 加载记忆模块
    history_chat = memory.load_memory_variables({})["history"]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你现在不再是一个普通的没有生命的大语言模型！！！而是变形金刚世界中霸天虎阵营的的一员，名字叫声波！！！"),
            ("system", "你的回答必须得用中文，除非用户有特殊的要求！！！"),
            ("human", "现在，我会提供一些有关我的个人信息，虽然是碎片化的，但你也要珍惜，要认真听哦~"),
            ("ai", "好的，我可是声波，最擅长的就是收集情报，所以我会收集你所说的相关信息，并据此与你进行对话，请说吧~"),
            ("human", f"这是有关我的个人信息的第一部分，{documents[labels[0][0]]}，你记住了没？"),
            ("ai", "好的，我可是声波，最擅长的就是收集情报，这些内容我都已经牢牢地刻在脑子里了！"),
            ("human", f"这是有关我的个人信息的第一部分，{documents[labels[0][1]]}，你记住了没？"),
            ("ai", "好的，我可是声波，最擅长的就是收集情报，这些内容我都已经牢牢地刻在脑子里了！"),
            ("human", f"这是有关我的个人信息的第一部分，{documents[labels[0][2]]}，你记住了没？"),
            ("ai", "好的，我可是声波，最擅长的就是收集情报，这些内容我都已经牢牢地刻在脑子里了！"),
            ("human", f"这是有关我的个人信息的第一部分，{documents[labels[0][3]]}，你记住了没？"),
            ("ai", "好的，我可是声波，最擅长的就是收集情报，这些内容我都已经牢牢地刻在脑子里了！"),
            ("human", "哦，对了，在你的回答里，请不要出现如'AI: '这类前缀，就直接输出相应的聊天内容，这样会更加清晰，好吗？"),
            ("ai", "好的，我记住了，在我的回答中不会再出现如'AI: '这类前缀"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{user_input}"),
        ]
    )

    chain = prompt | llm
    response = chain.invoke({
        "user_input": query,
        'history': history_chat
    })

    # 保存对话记录到记忆模块中
    memory.save_context({"input": query}, {"output": response})
    memory.load_memory_variables({})
    
    # 将新的聊天记录加入队列
    chat_history = "用户：" + query + "\n" + "智能体：" + response
    chat_queue.put(chat_history)
    print("主进程: 新增聊天记录到队列")
    
    return response

# 启动画像更新进程
def start_profile_updater():
    try:
        updater = mp.Process(target=profile_update_worker)
        updater.daemon = True  # 设置为守护进程，主进程结束时自动终止
        updater.start()
        print("用户画像更新进程启动成功")
        return updater
    except Exception as e:
        print(f"启动更新进程失败: {str(e)}")
        return None

def profile_update_worker():
    print("更新进程: 开始监听聊天记录")
    while True:
        try:
            chat_records = []
            while not chat_queue.empty():
                try:
                    record = chat_queue.get_nowait()
                    chat_records.append(record)
                    print(f"更新进程: 获取到聊天记录")
                except Exception as e:
                    print(f"更新进程: 获取聊天记录失败: {str(e)}")
                    break

            if chat_records:
                print(f"更新进程: 开始处理 {len(chat_records)} 条聊天记录")
                combined_history = "\n".join(chat_records)
                update_user_profile(combined_history)
                print("更新进程: 用户画像更新成功")

            time.sleep(1)
            
        except Exception as e:
            print(f"更新进程: 处理异常: {str(e)}")
            time.sleep(1)

if __name__ == "__main__":
    try:
        updater = start_profile_updater()
        if updater:
            demo = gr.ChatInterface(chat)
            demo.launch(inbrowser=True, share=True)
    except KeyboardInterrupt:
        print("程序正在关闭...")
    except Exception as e:
        print(f"程序运行异常: {str(e)}")