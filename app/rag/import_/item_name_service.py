from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import DataType

from app.infra.llm.providers import llm_provider
from app.infra.vectorstore import milvus_gateway
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import ITEM_NAME_CONTEXT_CHUNK_K, ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import step_log, logger


@step_log("validate_chunks_and_title")
def validate_chunks_and_title(state) -> tuple[list[dict],str]:
    # 1. 获取数据 chunks 和 file_title
    chunks = state.get("chunks")
    file_title = state.get("file_title")
    # 2. 非空判断
    if not chunks:
        logger.error("chunks内容为空，无法继续业务！")
        raise ValueError("chunks内容为空，无法继续业务！")
    if not file_title:
        file_title = chunks[0]['file_title'] or "default_file_title"
    # 3. 返回结束
    return chunks, file_title

@step_log("build_document_context")
def build_document_context(chunks) -> str:
    """
    进行上下文拼接
    """
    # 1. 截取top k chunk内容
    top_chunk = chunks[:ITEM_NAME_CONTEXT_CHUNK_K]
    # 2. 拼接上下文
    # 切片：1 标题：x 父标题：x 内容：x \n
    context = ""
    for index, chunk in enumerate(top_chunk, start=1):
        context += f"切片：{index} 标题：{chunk['title']} 父标题：{chunk['parent_title']} 内容：{chunk['content']} \n"
    # 3. 最大的字符串长度限制
    final_context = context[:ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS]
    return final_context

@step_log("recognize_item_name")
def recognize_item_name(context:str, file_title:str) -> str:
    # 1. 获取llm的客户端对象（llm/providers .chat()）
    chat_model = llm_provider.chat()
    # 2. 加载外部的提示词
    system_prompt_str = load_prompt("product_recognition_system")
    human_prompt_str = load_prompt(
        "item_name_recognition",
        file_title = file_title,
        context = context
    )
    # 3. 封装成我们提示词格式 HumanMessage SystemMessage
    messages = [
        SystemMessage(content = system_prompt_str),
        HumanMessage(content = human_prompt_str)
    ]
    # 4. 组装调用链
    chains = chat_model | StrOutputParser()
    # 5. 执行调用链获取item_name
    item_name = chains.invoke(messages)
    logger.info(f"调用模型进行item_name识别完毕！ item_name:{item_name}")
    # 6. 进行非空判断和兜底赋值
    if not item_name:
        item_name = file_title
    # 7. 返回item_name
    return item_name

@step_log("apply_item_name")
def apply_item_name(chunks: list[dict], item_name: str):
    """
        给chunks -> chunk -> item_name赋值
    """
    for chunk in chunks:
        chunk['item_name'] = item_name
    logger.info(f"完成chunks的item_name数据补充！{chunks[0]['item_name']}")

@step_log("embed_item_name")
def embed_item_name(item_name: str):
    """
        根据item_name生成稠密和稀疏向量
    """
    # 生成调用llm/probiders
    result = llm_provider.embed_documents([item_name])
    return result['dense'][0],result['sparse'][0]

@step_log("prepare_item_name_collection")
def prepare_item_name_collection():
    # item_name 存储的集合 一定创建么？

    # 1. 获取客户端对象
    milvus_client = milvus_gateway.client
    # 2. 判断集合是否存在
    if milvus_client.has_collection(collection_name=milvus_gateway.item_collection_name):
        # 存在
        logger.info(f"{milvus_gateway.item_collection_name}对应的集合存在，无需创建！")
        return
    # 3. 创建集合对应schema [field列]
    # 3.1 Create schema
    schema = milvus_client.create_schema(
        auto_id=True,
        enable_dynamic_field=True
    )
    # 3.2 Add fields to schema
    # https://milvus.io/docs/zh/v2.6.x/sparse_vector.md
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
    # 4. 创建集合对应indexes [索引]
    # 3.3 Prepare index parameters
    index_params = milvus_client.prepare_index_params()

    # 3.4 Add indexes
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        metric_type="COSINE",
        params = {
            "M": 64,
            "efConstruction": 100
        }
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE"}
    )
    # 5. 创建集合（集合的名字 schema indexes）
    milvus_client.create_collection(
        collection_name=milvus_gateway.item_collection_name,
        schema=schema,
        index_params=index_params
    )
    logger.info(f"{milvus_gateway.item_collection_name}第一次完成初始化！")

@step_log("upsert_item_name")
def upsert_item_name(item_name: str, file_title: str, dense_vector: list[float], sparse_vector: dict[int, float]):
    """
        先删除 / 再插入 幂等性
    """
    milvus_client = milvus_gateway.client
    # 1. 先根据file_title删除
    milvus_client.delete(
        collection_name=milvus_gateway.item_collection_name,
        filter=f"file_title == '{file_title}'"
    )
    # 2. 插入新的数据即可
    result = milvus_client.insert(
        collection_name=milvus_gateway.item_collection_name,
        data=[{
            "item_name": item_name,
            "file_title": file_title,
            "dense_vector": dense_vector,
            "sparse_vector": sparse_vector
        }]
    )
    logger.info(f"{item_name}对应的数据已经插入到{milvus_gateway.item_collection_name}对应的集合中！返回结果：{result}")

# 主业务入口
@step_log("recognize_and_index_item_name")
def recognize_and_index_item_name(state: ImportGraphState) -> ImportGraphState:
    """
    主体识别服务：
    1. 基于 chunks 构造上下文
    2. 调用 LLM 识别 item_name
    3. 将 item_name 回填到 state 和 chunks
    4. 同步写入主体名称索引
    """
    # 1. 进行参数校验
    chunks, file_title = validate_chunks_and_title(state)
    # 2. 进行上下文的拼接 chunks
    context = build_document_context(chunks)
    # 3. 进行 item_name 的识别了 llm
    item_name = recognize_item_name(context, file_title)
    # 4. 修改所有chunks的item_name属性
    apply_item_name(chunks, item_name)
    # 5. 对item_name进行向量化，生成稠密和稀疏向量
    dense_vector, sparse_vector = embed_item_name(item_name)
    # 6. 准备item_name对应的集合信息
    prepare_item_name_collection()
    # 7. 更新或者存储item_name到对应的集合
    upsert_item_name(item_name, file_title, dense_vector, sparse_vector)
    # 8. 更新state数据
    state['chunks'] = chunks
    state['item_name'] = item_name
    return state