from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger,step_log
from app.infra.llm.providers import llm_provider
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.infra.llm.providers import llm_provider


def get_data_and_validates(state:QueryGraphState) -> tuple[str,list[str]]:
    """
    获取参数和校验
    :param state:
    :return: 重写问题以及关联的item_names
    """
    rewritten_query = state.get("rewritten_query")
    item_names = state.get("item_names",[])

    if not rewritten_query or len(item_names) == 0:
        logger.error(f"重写问题或者关联的主体为空,无法继续业务!")
        raise ValueError(f"重写问题或者关联的主体为空,无法继续业务!")

    return rewritten_query, item_names


def milvus_search_hyde_entity(hyde_answer:str, rewritten_query:str, item_names:list[str]):
    """
      使用重写问题，对向量库进行搜索！
      注意：需要添加item_name的过滤条件
    :param rewritten_query:
    :param item_names:
    :return: 返回处理一层后的列表
    """
    # 1. 重写问题生成向量
    embedding_result = llm_provider.embed_documents([rewritten_query+":"+hyde_answer])
    dense_vector = embedding_result['dense'][0]
    sparse_vector = embedding_result['sparse'][0]
    # 2. 创建annSearchRequest
    ann_reqs = milvus_gateway.create_requests(dense_vector=dense_vector,
                                              sparse_vector=sparse_vector, expr=f"item_name in {item_names}", limit=5*2)
    # 3. 调用混合检索（设置输出列）
    milvus_result = milvus_gateway.hybrid_search(
        collection_name=milvus_gateway.chunk_collection_name,
        reqs=ann_reqs,
        ranker_weights=(0.6,0.4),
        limit=5,
        norm_score=True,
        output_fields=[
            "chunk_id",
            "title",
            "parent_title",
            "file_title",
            "item_name",
            "content",
            "part"
        ]
    )
    # 4. 返回第一层结果
    return milvus_result[0] if milvus_result and len(milvus_result) > 0 else []


def normalize_retrieved_chunk(milvus_response: list[dict]) -> list[dict]:
    final_list_dict = []
    for milvus_dict in milvus_response:
        # milvus_dict {id, distance, entity : {} }
        entity = milvus_dict.get("entity",{})

        final_list_dict.append(
            {
                "chunk_id": milvus_dict.get("id") or entity.get("chunk_id"),  # 片段ID
                "item_name": entity.get("item_name", ""),  # 归属主体名称
                "title": entity.get("title"),  # 片段标题
                "parent_title": entity.get("parent_title"),  # 父标题/章节
                "part": entity.get("part"),  # 部分标识
                "file_title": entity.get("file_title"),  # 来源文件标题
                "content": entity.get("content", ""),  # 片段文本内容
                "score": milvus_dict.get("distance", 0.0),  # 相似度分数
                "type": "milvus",  # 来源类型（向量库）
                "url": None,  # 附件URL（无）
            }
        )
        return final_list_dict


def call_llm_by_rewritten_query(rewritten_query) -> str:
    """
      调用模型生成假设性答案
    :param rewritten_query:
    :return:
    """
    # 1. 获取模型对象（视觉模型 还是 大语言模型（json模式））
    llm_client = llm_provider.chat()
    # 2. 加载和封装提示词
    prompt_text = load_prompt("hyde_prompt", rewritten_query = rewritten_query)
    messages = [
        HumanMessage(content=prompt_text)
    ]
    # 3. 封装调用链
    chains = llm_client | StrOutputParser()
    # 4. 执行并获取返回结果
    hyde_answer = chains.invoke(messages)
    # 5. 返回
    return hyde_answer



def search_by_hyde(state: QueryGraphState):
    """
    向量检索服务：
    1. 根据改写后的问题和限定的商品范围
    2. 利用 BGEM3 混合检索（稠密+稀疏）计数
    3. 从 Milvus 向量数据库中召回 Top-k 最相关的知识切片
    4. 回写 embedding_chunks
    """
    # 1. 参数获取和校验
    rewritten_query, item_names = get_data_and_validates(state)
    # 2. 根据问题获取答案(lm)
    hyde_answer = call_llm_by_rewritten_query(rewritten_query)

    # 3. 进行向量库混合内容检索
    milvus_response = milvus_search_hyde_entity(hyde_answer,rewritten_query,item_names)
    # 4. 进行数据格式化处理
    #  [dict {id , distance , entity : {} } -> 目标格式  {}]
    final_list_dict = normalize_retrieved_chunk(milvus_response)
    # 5. 直接返回数据 todo: 修改节点即可
    return final_list_dict