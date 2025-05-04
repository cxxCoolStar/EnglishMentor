import json
import os
import time
from datetime import datetime, date, timedelta
from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from .session_history import get_session_history
from utils.logger import LOG

# 获取项目根目录路径（假设src在项目根目录下）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ApiRateLimiter:
    """
    API调用次数限制器，用于控制每天的API调用次数
    """

    def __init__(self, limit, storage_path=None):
        """
        初始化API调用次数限制器

        参数:
            limit (int): 每天最大调用次数
            storage_path (str, optional): 存储调用计数的文件路径
        """
        self.limit = limit
        self.storage_path = storage_path or os.path.join(PROJECT_ROOT, "data", "api_usage.json")
        # 确保存储目录存在
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        self._load_or_initialize_counter()

    def _load_or_initialize_counter(self):
        """
        加载或初始化API调用计数器
        """
        today = str(date.today())
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    # 如果是新的一天，重置计数器
                    if data.get("date") != today:
                        self.count = 0
                        self.date = today
                    else:
                        self.count = data.get("count", 0)
                        self.date = data.get("date")
            else:
                self.count = 0
                self.date = today
                self._save_counter()
        except (json.JSONDecodeError, FileNotFoundError):
            # 如果文件不存在或格式错误，重置计数器
            self.count = 0
            self.date = today
            self._save_counter()

    def _save_counter(self):
        """
        保存API调用计数到文件
        """
        data = {
            "date": self.date,
            "count": self.count
        }
        with open(self.storage_path, "w", encoding="utf-8") as file:
            json.dump(data, file)

    def increment(self):
        """
        增加调用计数并保存

        返回:
            bool: 是否成功（未超过限制）
        """
        # 检查是否是新的一天
        today = str(date.today())
        if self.date != today:
            self.count = 0
            self.date = today

        # 检查是否超过限制
        if self.count >= self.limit:
            return False

        # 增加计数并保存
        self.count += 1
        self._save_counter()
        return True

    def get_remaining(self):
        """
        获取今天剩余的API调用次数

        返回:
            int: 剩余调用次数
        """
        # 检查是否是新的一天
        today = str(date.today())
        if self.date != today:
            return self.limit

        return max(0, self.limit - self.count)


class AgentBase(ABC):
    """
    抽象基类，提供代理的共有功能。
    包含API调用限制功能，每天最多调用50次。
    """
    # 类变量，所有实例共享同一个API限制器
    _api_limiter = ApiRateLimiter(limit=50)

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

    def check_api_limit(self):
        """
        检查API调用限制，如果超过限制，抛出异常

        返回:
            bool: 是否可以继续调用API
        """
        if not AgentBase._api_limiter.increment():
            remaining_time = self._get_time_until_tomorrow()
            LOG.warning(f"达到今日API调用上限(50次)，请等待{remaining_time}后再试")
            raise RateLimitExceededError(f"达到今日API调用上限(50次)，请等待{remaining_time}后再试")
        return True

    def _get_time_until_tomorrow(self):
        """
        计算到明天0点的剩余时间

        返回:
            str: 格式化的剩余时间字符串
        """
        now = datetime.now()
        tomorrow = datetime(now.year, now.month, now.day) + timedelta(days=1)
        delta = tomorrow - now

        hours, remainder = divmod(delta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        return f"{hours}小时{minutes}分钟"

    def get_remaining_calls(self):
        """
        获取今日剩余API调用次数

        返回:
            int: 剩余调用次数
        """
        return AgentBase._api_limiter.get_remaining()

    def chat_with_history(self, user_input, session_id=None):
        """
        处理用户输入，生成包含聊天历史的回复。
        在调用API前会检查调用限制。

        参数:
            user_input (str): 用户输入的消息
            session_id (str, optional): 会话的唯一标识符

        返回:
            str: AI 生成的回复
        """
        if session_id is None:
            session_id = self.session_id

        # 检查API调用限制
        try:
            self.check_api_limit()
        except RateLimitExceededError as e:
            return f"抱歉，{str(e)}"

        # 调用API生成回复
        response = self.chatbot_with_history.invoke(
            [HumanMessage(content=user_input)],  # 将用户输入封装为 HumanMessage
            {"configurable": {"session_id": session_id}},  # 传入配置，包括会话ID
        )

        LOG.debug(f"[ChatBot][{self.name}] {response.content}")  # 记录调试日志
        return response.content  # 返回生成的回复内容


class RateLimitExceededError(Exception):
    """
    API调用次数超过限制时抛出的异常
    """
    pass