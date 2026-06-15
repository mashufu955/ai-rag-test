from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger,step_log
from app.infra.llm.providers import llm_provider
from app.infra.vectorstore.milvus_gateway import milvus_gateway


def get_data_and_validates(state:QueryGraphState) -> tuple[str,list[str]]:
    """
    获取参数和校验
    :param state:
    :return: 重写问题以及关联的item_names
    """
    rewritten_query = state.get("rewritten_query")
    item_names = state.get("item_names", [])

    if not rewritten_query or len(item_names) == 0:
        logger.error(f"重写问题或者关联的主体为空，无法继续业务！")
        raise ValueError(f"重写问题或者关联的主体为空，无法继续业务！")

    return rewritten_query, item_names


def milvus_search_entity(rewritten_query, item_names):
    """
        使用重写问题，对向量库进行搜索！
        注意：需要添加item_name的过滤条件
    :param rewritten_query:
    :param item_names:
    :return: 返回处理一层后的列表
    """
    # 1. 重写问题生成向量
    embedding_result = llm_provider.embed_documents([rewritten_query])
    dense_vector = embedding_result['dense'][0]
    sparse_vector = embedding_result['sparse'][0]
    # 2. 创建annSearchRequest
    ann_reqs = milvus_gateway.create_requests(dense_vector=dense_vector,
                                              #  item_name in ['xx','xxx','xxx']
                                              #  item_name in [1,2,3,4]
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
        # milvus_dict {id, distance, entity : {}}
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



def search_by_embedding(state: QueryGraphState):
    """
    向量检索服务：
    1. 根据改写后的问题和限定的商品范围
    2. 利用 BGEM3 混合检索（稠密+稀疏）技术
    3. 从 Milvus 向量数据库中召回 Top-k 最相关的知识切片
    4. 回写 embedding_chunks
    """
    # 1. 参数获取和校验
    rewritten_query, item_names = get_data_and_validates(state)
    # 2. 进行向量库混合内容检索
    milvus_response = milvus_search_entity(rewritten_query,item_names)
    # 3. 进行数据格式化处理
    #  [dict {id , distance , entity : {} } -> 目标格式  {}]
    final_list_dict = normalize_retrieved_chunk(milvus_response)
    # 4. 直接返回数据 todo: 修改节点即可
    return final_list_dict