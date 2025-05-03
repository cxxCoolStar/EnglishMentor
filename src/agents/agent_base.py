import json
from abc import ABC, abstractmethod

from langchain_openai import ChatOpenAI  # 更改为 OpenAI 兼容接口
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # 导入提示模板相关类
from langchain_core.messages import HumanMessage  # 导入消息类
from langchain_core.runnables.history import RunnableWithMessageHistory  # 导入带有消息历史的可运行类

from .session_history import get_session_history  # 导入会话历史相关方法
from utils.logger import LOG  # 导入日志工具
import os

# 获取项目根目录路径（假设src在项目根目录下）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class AgentBase(ABC):
    """
    抽象基类，提供代理的共有功能。
    """

    def __init__(self, name, prompt_file, intro_file=None, session_id=None):
        self.name = name
        # 将相对路径转换为绝对路径
        self.prompt_file = os.path.join(PROJECT_ROOT, prompt_file)
        self.intro_file = os.path.join(PROJECT_ROOT, intro_file) if intro_file else None
        self.session_id = session_id if session_id else self.name
        self.prompt = self.load_prompt()
        self.intro_messages = self.load_intro() if self.intro_file else []
        self.create_chatbot()

    def load_prompt(self):
        """
        从文件加载系统提示语。
        """
        try:
            with open(self.prompt_file, "r", encoding="utf-8") as file:
                return file.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到提示文件 {self.prompt_file}!")

    def load_intro(self):
        """
        从 JSON 文件加载初始消息。
        """
        try:
            with open(self.intro_file, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到初始消息文件 {self.intro_file}!")
        except json.JSONDecodeError:
            raise ValueError(f"初始消息文件 {self.intro_file} 包含无效的 JSON!")

    def create_chatbot(self):
        """
        初始化聊天机器人，包括系统提示和消息历史记录。
        """
        # 创建聊天提示模板，包括系统提示和消息占位符
        system_prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt),  # 系统提示部分
            MessagesPlaceholder(variable_name="messages"),  # 消息占位符
        ])

        # 从环境变量获取API密钥
        api_key = os.getenv("DASHSCOPE_API_KEY").strip()
        if not api_key:
            LOG.warning("环境变量 DASHSCOPE_API_KEY 未设置，请设置API密钥")
            raise ValueError("API密钥未设置，请设置环境变量 DASHSCOPE_API_KEY")

        # 初始化 ChatOpenAI 模型，配置模型参数
        self.chatbot = system_prompt | ChatOpenAI(
            api_key=api_key,  # 从环境变量获取API密钥
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云DashScope API
            model="qwen-plus",  # 使用千问Plus模型
            max_tokens=8192,  # 最大生成的token数
            temperature=0.8,  # 生成文本的随机性
        )

        # 将聊天机器人与消息历史记录关联
        self.chatbot_with_history = RunnableWithMessageHistory(self.chatbot, get_session_history)

    def chat_with_history(self, user_input, session_id=None):
        """
        处理用户输入，生成包含聊天历史的回复。

        参数:
            user_input (str): 用户输入的消息
            session_id (str, optional): 会话的唯一标识符

        返回:
            str: AI 生成的回复
        """
        if session_id is None:
            session_id = self.session_id

        response = self.chatbot_with_history.invoke(
            [HumanMessage(content=user_input)],  # 将用户输入封装为 HumanMessage
            {"configurable": {"session_id": session_id}},  # 传入配置，包括会话ID
        )

        LOG.debug(f"[ChatBot][{self.name}] {response.content}")  # 记录调试日志
        return response.content  # 返回生成的回复内容
