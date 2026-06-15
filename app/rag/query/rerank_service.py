from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.infra.llm.providers import llm_provider
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import step_log, logger
from app.rag.query.config import RERANK_MAX_INPUT_TOKENS, RERANK_SUMMARY_CHAR_RATIO, RERANK_MIN_SUMMARY_CHARS, \
    RERANK_MAX_TOPK, RERANK_MIN_TOPK, RERANK_GAP_RATIO, RERANK_GAP_ABS


def get_rewritten_query_and_validate(state):
    """
    获取和校验核心参数
    :param state:
    :return:
    """
    # 获取参数
    rrf_chunks = state.get("rrf_chunks",[])
    web_search_docs = state.get("web_search_docs",[])
    rewritten_query = state.get("rewritten_query")

    # 非空判定
    if len(rrf_chunks) == 0 or len(web_search_docs) == 0 or not rewritten_query:
        logger.error(f"关键参数为空,业务无法继续进行!!")
        raise ValueError(f"关键参数为空,业务无法继续进行!!")

    return rrf_chunks, web_search_docs, rewritten_query


def deal_chunk_list(rrf_chunks, web_search_docs):
    """
    进行两路数据融合
        rrf_chunks ->  chunk_id title parent_title part file_title content type url item_name
        web_search_docs -> snippet title  url
    融合目标 -> 大模型提供数据支撑
        title
        text
        type  milvus / web  ->  数据来源
        url   milvus - none / web -> url (资源的网页 | 资源关联的图)
        score milvus 排名的分 / web 没有分 ->  reranker的分
    :param rrf_chunks:
    :param web_search_docs:
    :return:
    """
    final_chunk_list = []
    # 1. 循环rrf_chunks
    for chunk in rrf_chunks:
        final_chunk_list.append({
            "title": chunk.get("title"),
            "text": chunk.get("content"),
            "type": "milvus",
            "url": None,
            "score": 0.0
        })
    # 2. 循环web_search_docs
    for doc in web_search_docs:
        final_chunk_list.append({
            "title": doc.get("title"),
            "text": doc.get("snippet"),
            "type": "web",
            "url": doc.get("url"),
            "score": 0.0
        })

    # 3. 返回结果
    return final_chunk_list


def ranker_answer_llm_deal(answer:str, limit:int, question:str) -> str:
    """
     做问题的回答压缩！
    :param answer:
    :param limit:
    :param question:
    :return:
    """
    # 1. 加载模型客户端
    chat_client = llm_provider.chat()
    # 2. 加载提示词
    prompt_text = load_prompt("reranker_text_refine", question=question,answer=answer,limit=limit)
    # 3. 构建messages
    messages = [
        HumanMessage(
            content=prompt_text
        )
    ]
    # 4. 构建调用链
    chain = chat_client | StrOutputParser()
    # 5. 执行
    refine_answer = chain.invoke(messages)
    return refine_answer


def deal_question_answer_pair_list(rewritten_query, final_chunk_list) -> list[list[str]]:
    """
    生成问题和答案对列表！超长，调用 ranker_answer_llm_deal
    :param rewritten_query:
    :param final_chunk_list:
    :return:
    """
    question_answer_pair_list = []
    # 1. 检查问题的长度 rewritten_query
    reranker_model = llm_provider.reranker_model()
    tokenizer = reranker_model.tokenizer
    # add_special_tokens = False单纯算我这个字符串对应token列表! 不用关注我前后的特殊字符
    # [1,2,3,55,66]
    question_token_ids_list = tokenizer.encode(rewritten_query, add_special_tokens=False)
    question_token_len = len(question_token_ids_list)
    # 2. 循环答案final_chunk_list
    for chunk in final_chunk_list:
        current_answer = chunk.get("text","")
        # 3. 检查答案的长度
        current_answer_token_len = len(tokenizer.encode(current_answer, add_special_tokens=False))
        # 4. 超长就进行模型压缩
        if current_answer_token_len + question_token_len + 4 > RERANK_MAX_INPUT_TOKENS:
            # 计算字符串长度
            limit = max(
                RERANK_MIN_SUMMARY_CHARS,
                # 转成整数
                int((RERANK_MAX_INPUT_TOKENS - question_token_len - 4) / RERANK_SUMMARY_CHAR_RATIO)
            )
            # 调用模型
            current_answer = ranker_answer_llm_deal(answer=current_answer,limit=limit,question=rewritten_query)
        # 5. 添加到列表中即可
        question_answer_pair_list.append([rewritten_query,current_answer])
    # 6. 返回结果
    return question_answer_pair_list


def reranker_score_pair_list(question_answer_pair_list):
    """
    调用reranker模型进行打分
    :param question_answer_pair_list:
    :return:
    """
    reranker_model = llm_provider.reranker_model()
    # normalize=True 归一化 将分值拉到 0-1 之间！方便进行后续算法统计！！
    score_list = reranker_model.compute_score(question_answer_pair_list, normalize=True)
    logger.info(f"完成对数据：{question_answer_pair_list}的打分，分数为：{score_list}")
    return score_list


def sort_final_chunk_list(final_chunk_list, scores_list):
    """
    数据进行排序
    :param final_chunk_list:
    :param scores_list:
    :return:
    """
    # 数据融合 分 -> final_chunk_list
    for chunk, score in zip(final_chunk_list, scores_list):
        chunk['score'] = score

    # 获得是带有打分的列表数据 没有排序
    logger.info(f"没排序前的顺序:{final_chunk_list}")
    logger.info("*" * 60)
    final_chunk_list.sort(key=lambda x: x['score'], reverse=True)
    logger.info(f"排序后的顺序:{final_chunk_list}")
    return final_chunk_list


def dynamic_cut_chunk_list(final_chunk_list) -> list[dict]:
    """
    动态截取数据
    :param final_chunk_list:
    :return:
    """
    max_number = RERANK_MAX_TOPK
    min_number = RERANK_MIN_TOPK
    gap_abs    = RERANK_GAP_ABS
    gap_ratio  = RERANK_GAP_RATIO
    # 处理max_number大于列表长度的可能
    max_number = min(max_number,len(final_chunk_list))
    # 声明top k并赋值 max_number 没有断崖,默认给截取全部!
    top_k = max_number
    # 循环的目标寻找断崖
    # 有可能min > max
    if max_number > min_number:
        # 起始位置 最小的位置的下标
        # 结束位置 最大的位置的下标 - 1  (前一个)  == max_number - 1 - 1 前一个
        for index in range(min_number-1,max_number-1):
            score_1 = final_chunk_list[index].get('score',0.0)
            score_2 = final_chunk_list[index+1].get('score',0.0)
            abs_score = score_1 - score_2
            ratio_score = abs_score / (score_1 + 1e-7)
            # 断崖判断
            if abs_score > gap_abs or ratio_score > gap_ratio:
                top_k = index + 1
                break
    logger.info(f"已经完成断崖数据截取,进入数量:{len(final_chunk_list)},截取数据:{top_k}")
    # 截取数据
    return final_chunk_list[:top_k]


def rerank_documents(state: QueryGraphState) -> list[dict]:
    """
    重排节点主入口
    流程：校验输入 → 合并本地+网页 → 模型打分排序 → 动态截断
    输出最终高质量候选文档列表
    """
    # 1. 获取数据并且校验
    rrf_chunks, web_search_docs, rewritten_query = get_rewritten_query_and_validate(state)

    # 2. 多路数据融合格式统一
    final_chunk_list = deal_chunk_list(rrf_chunks,web_search_docs)

    # 3. 生成问题和答案列表[[问题,答案],[]]
    question_answer_pair_list = deal_question_answer_pair_list(rewritten_query,final_chunk_list)

    # 4. reranker模型进行打分
    # scores_list -> [1,2,3,4,5,6,7] -> question_answer_pair_list (优化后的) -> final_chunk_list
    scores_list = reranker_score_pair_list(question_answer_pair_list)

    # 5. 原始数据进行赋分和排序
    final_chunk_list = sort_final_chunk_list(final_chunk_list,scores_list)

    # 6. 动态截取数据 topk
    final_chunk_list = dynamic_cut_chunk_list(final_chunk_list)

    state['reranked_docs'] = final_chunk_list
    return state