import sys
import time

from app.shared.runtime.logger import node_log
from app.rag.query.embedding_search_service import search_by_embedding
from app.shared.utils.task_utils import add_done_task, add_running_task

@node_log("node_search_embedding")
def node_search_embedding(state):
    """
    节点功能：进行向量内容检索
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    time.sleep(0.5)
    embedding_chunks = search_by_embedding(state)
    
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {
        "embedding_chunks": embedding_chunks
    }