import sys
import time

from app.shared.runtime.logger import node_log
from app.rag.query.hyde_search_service import search_by_hyde
from app.shared.utils.task_utils import add_done_task, add_running_task

@node_log("node_search_embedding_hyde")
def node_search_embedding_hyde(state):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    time.sleep(0.5)
    hyde_embedding_chunks = search_by_hyde(state)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {
        "hyde_embedding_chunks":hyde_embedding_chunks
    }