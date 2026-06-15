from app.process.query.agent.state import QueryGraphState
from app.shared.utils.task_utils import add_done_task, add_running_task, push_to_session
from app.shared.utils.sse_utils import SSEEvent
from app.shared.runtime.logger import logger
import time
import sys

def generate_answer(state: QueryGraphState) -> QueryGraphState:
    """
    答案生成服务：
    1. 检查前置答案（如有追问或拒绝回答，直接输出）
    2. 构建 Prompt（用户问题 + 历史对话 + Topk 文档）
    3. 调用 LLM 生成最终答案（支持流式推送）
    4. 从引用文档中提取图片 URL
    5. 写入 MongoDB 历史记录
    6. 回写 answer 和 image_urls
    """

    print("---node_answer_output 节点处理开始---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    session_id = state["session_id"]
    is_stream = state.get("is_stream", True)
    base_answer = state.get("answer") or f"这是关于 {state.get("original_query", "当前问题")} 的测试回答，正在演示打字机流式输出效果。"
    final_text = ""

    if is_stream:
        for ch in base_answer:
            final_text += ch
            push_to_session(session_id, SSEEvent, {"delta":ch})
            time.sleep(0.05)

        image_urls = ["https://example.com/demo-1.png", "https://example.com/demo-2.png"]
        push_to_session(
            session_id,
            SSEEvent,
            {
                "answer": final_text,
                "status": "completed",
                "image_urls": image_urls
            }
        )
        logger.info(f"流失输出完成，总长度：{len(final_text)}")
    else:
        final_text = base_answer

    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    print("---node_answer_output 节点处理结束---")
    # 关键点：return 必须保留 session_id!
    return {
        "session_id": session_id,   # 必须带回去
        "answer": "你的回答内容",
        "is_stream": state.get("is_stream")
    }