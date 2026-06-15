import sys
import time

from app.shared.runtime.logger import node_log
from app.rag.query.rerank_service import rerank_documents
from app.shared.utils.task_utils import add_done_task, add_running_task

@node_log("node_rerank")
def node_rerank(state):
    """
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    time.sleep(0.5)
    state = rerank_documents(state)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    return state


if __name__ == "__main__":
    mock_rrf_chunks = [
        {"chunk_id": "local_1", "content": "RRF是一种倒数排名融合算法", "title": "算法介绍", "type": "milvus"},
        {"chunk_id": "local_2", "content": "二大爷最亲!", "title": "模型介绍","type":"milvus"},
    ]
    mock_web_docs = [
        {"title": "Rerank技术详解", "url": "http://web.com/1", "snippet": "Rerank即重排序，常用于RAG系统的第二阶段"},
        {"title": "Rerank技术详解", "url": "http://web.com/1", "snippet": "叔比舅更亲?"},
        {"title": "Rerank技术详解", "url": "http://web.com/1", "snippet": "姨比姑更亲?"}
    ]
    mock_state = {
        "session_id": "test_rerank_session",
        "rewritten_query": "请问哪个亲戚更亲？",
        "rrf_chunks": mock_rrf_chunks,
        "web_search_docs": mock_web_docs,
        "is_stream": False,
    }
    result = node_rerank(mock_state)
    print(result)