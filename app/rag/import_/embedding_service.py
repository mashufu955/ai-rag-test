from app.infra.llm.providers import llm_provider
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import EMBEDDING_BATCH_SIZE
from app.shared.runtime.logger import step_log, logger


# 1. 校验chunks
@step_log("require_chunks")
def require_chunks(state) -> list[dict]:
    # 1. 获取chunks
    chunks = state.get("chunks")
    # 2. 非空校验
    if not chunks or len(chunks) == 0:
        logger.error(f"chunks数据被置空，无法继续业务！")
        raise ValueError(f"chunks数据被置空，无法继续业务！")
    return chunks

# 2. 批量生成向量 / 预设值 5个一批
# 注意： 生成向量 item_name + content
# * 后面的必须指定名称传递！
@step_log("embed_chunks")
def embed_chunks(chunks:list[dict], *, step: int = EMBEDDING_BATCH_SIZE) -> list[dict]:
    """
        批量生成chunk对应的向量！
        注意： item_name+content
    """
    final_chunks = []
    # 1 2 3 4 5     6 7 8 9 10      11 12
    # 1. 分批循环获取chunks内容
    for index in range(0, len(chunks), step):
        # index 0  5  10
        # 2. 当前批次
        step_chunks = chunks[index: index + step]
        # 3. 组装生成向量的字符串列表
        step_vector_list = []
        # 4. 处理要生成向量的字符串
        for current_chunk in step_chunks:
            # 匹配的规则 item_name + content
            step_vector_list.append(
                f"主体名：{current_chunk['item_name']},内容：{current_chunk['content']}"
            )
        # 5. 批量生成向量 [1,2,3,4,5]
        result = llm_provider.embed_documents(step_vector_list)
        """
            result = {
                "dense":[[],[],[],[],[]],
                "sparse":[{},{},{},{},{}]
            }
        """
        # 6. 循环获取向量创建一个新的chunk添加到final_chunks
        for index, chunk in enumerate(step_chunks):
            # 浅copy 不copy嵌套内容
            # item_name content ... 向量 = []
            chunk_new = chunk.copy()
            chunk_new['dense_vector'] = result["dense"][index]
            chunk_new['sparse_vector'] = result["sparse"][index]
            final_chunks.append(chunk_new)
    # 7. 返回结果
    logger.info(f"已经完成chunks向量化：原始数据：{chunks[0]} 向量后：{final_chunks[0]}")
    return final_chunks

@step_log("generate_chunk_embeddings")
def generate_chunk_embeddings(state: ImportGraphState) -> ImportGraphState:
    """
    向量化服务：
    1. 读取 chunks
    2. 生成 dense_vector / sparse_vector
    3. 将向量结果补充回 chunks
    """
    chunks = require_chunks(state)
    # 带有向量
    final_chunks = embed_chunks(chunks)
    # 修改state
    state['chunks'] = final_chunks
    return state