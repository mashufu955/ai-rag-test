"""
模型提供者模块，统一封装聊天模型、Embedding 与 Reranker 的访问方式。
"""
from langchain_openai import ChatOpenAI

from app.infra.config import infra_config
from app.shared.model import generate_embeddings, get_bge_m3_ef, get_llm_client, get_reranker_model


class LLMProvider:
    def chat(self, model: str | None = None, json_mode: bool = False) -> ChatOpenAI:
        """
        获取聊天模型客户端。

        Args:
            model: 可选模型名；为空时使用默认聊天模型。
            json_mode: 是否启用 JSON 输出模式，适用于结构化抽取场景。

        Returns:
            ChatOpenAI: 可直接调用或流式调用的聊天模型客户端。
        """
        return get_llm_client(model=model, json_mode=json_mode)

    def vision_chat(self) -> ChatOpenAI:
        """
        获取视觉模型客户端。

        Returns:
            ChatOpenAI: 面向图片理解场景的视觉模型客户端。
        """
        return get_llm_client(model=infra_config.llm.lv_model)

    def embedding_model(self):
        """
        获取 Embedding 模型对象。

        Returns:
            Any: BGE-M3 Embedding 模型实例。
        """
        return get_bge_m3_ef()

    def reranker_model(self):
        """
        获取重排模型对象。
        Returns:
            Any: 可对问答对进行相关性打分的重排模型实例。
        """
        return get_reranker_model()

    def embed_documents(self, texts: list[str]) -> dict:
        """
        为文本列表生成向量表示。

        Args:
            texts: 待向量化的文本列表。

        Returns:
            dict: 同时包含稠密向量与稀疏向量的结果字典。
        """
        return generate_embeddings(texts)


llm_provider = LLMProvider()
