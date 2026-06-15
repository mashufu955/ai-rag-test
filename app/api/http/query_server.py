from mimetypes import guess_type
from pathlib import Path
import sys
import uuid

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware

from app.api.schema.query_schema import QueryRequestParam, QueryStreamResponse, QueryNotStreamResponse,HistoryCleanResponse,HistoryResponse,HistoryItemResponse
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.infra.config.providers import settings
from app.process.query.agent.main_graph import query_graph_app
from app.process.query.agent.state import create_query_default_state, QueryGraphState
from app.shared.utils.sse_utils import SSEEvent, create_sse_queue, push_to_session, sse_generator
from app.shared.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    clear_task,
    get_done_task_list,
    get_task_result,
    update_task_status
)

from app.infra.persistence.history_repository import history_repository

# 定义fastapi对象
app = FastAPI(
    title=settings.query_app_name,
    description="描述，进行rag查询的服务对象",
    version="0.2.0"
)

# 跨域处理
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # 198.1.1.1 , 192.168.1.1
    allow_methods=['*'],  # get post  put ...
    allow_headers=['*']  # content-type ...
)

# 1. 返回chat对应的页面
# /html get
@app.get("/html")
def chat_html():

    chat_html_path_obj = PROJECT_ROOT / "app" / "resources" / "html" / "chat.html"

    return FileResponse(
        path = chat_html_path_obj,
        media_type=guess_type(chat_html_path_obj.name)[0]
    )

# 2. /health get 健康检查接口
@app.get("/health")
def health():
    return {
        "code": 200,
        "message": "可以访问！！"
    }

# 3. /stream/{session_id} get
@app.get("/stream/{session_id}")
def stream(session_id, request:Request):
    return StreamingResponse(
        sse_generator(session_id,request),
        media_type="text/event-stream"
    )

def invoke_query_graph(session_id:str, query:str, is_stream:bool=False):
    # 执行 动态测试
    state = create_query_default_state(
        session_id=session_id,
        original_query=query,
        is_stream=is_stream
    )
    # 创建一个队列 session_id <-- 数据

    # 清空task_utils的数据
    clear_task(session_id)

    if is_stream:
        create_sse_queue(session_id)

    try:
        update_task_status(session_id, TASK_STATUS_PROCESSING,is_stream)
        logger.info(f"开始执行，执行参数为：{state}")
        result_state = query_graph_app.invoke(state)
        logger.info(f"执行结束，执行结果为：{result_state}")
        update_task_status(session_id, TASK_STATUS_COMPLETED,is_stream)

        image_urls = ["http://www.baidu.com/img/bd_logo.png",
                      "https://gips3.baidu.com/it/u=1821127123,1149655687&fm=3028&app=3028&f=JPEG&fmt=auto?w=720&h=1280"]
        push_to_session(
            session_id,
            SSEEvent.FINAL, # 显示图片
            {
                "answer": result_state['answer'],
                "status": "completed",
                "image_urls": image_urls
            }
        )
        # 返回结果！非流式需要
        return result_state
    except Exception as e:
        update_task_status(session_id,TASK_STATUS_FAILED,is_stream)
        push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})
        logger.exception(f"{session_id}执行出现了异常!!")


# 4. 查询和提问接口
# /query post {} | {}
@app.post("/query")
def query(backgroundtasks: BackgroundTasks, request: QueryRequestParam):
    """
        1. 获取stream状态
        2. true 异步 后台执行 图的调用过程 backgroundtask
           异步的返回结果
        3. false 同步 直接调用
           同步的返回结果
    :param request:
    :return:
    """
    session_id = request.session_id or str(uuid.uuid4())
    is_stream = request.is_stream
    query = request.query

    # 是否异步
    if is_stream:
        # 异步执行
        backgroundtasks.add_task(invoke_query_graph,
                                 session_id=session_id,
                                 query=query,
                                 is_stream=is_stream)
        # 立即向下
        return QueryStreamResponse(
            message=f"开启：{session_id}异步任务执行!",
            session_id=session_id
        )
    else:
        # 同步执行 死等
        final_state: QueryGraphState = invoke_query_graph(
            session_id=session_id,
            query=query,
            is_stream=is_stream
        )
        return QueryNotStreamResponse(
            message=f"{session_id}对应的任务已经处理完毕！！",
            session_id=session_id,
            answer=final_state.get("answer"),
            done_list=get_done_task_list(session_id),
            image_urls=final_state.get("image_urls")
        )


# 清空历史对话记录
@app.delete("/history/{session_id}")
def remove_history(session_id:str):
    delete_count = history_repository.clear_session(session_id=session_id)
    logger.info(f"清空：{session_id}对应的历史记录！清空数量：{delete_count}")
    return HistoryCleanResponse(
        message=f"清空：{session_id}对应的历史记录！清空数量：{delete_count}",
        deleted_count=delete_count
    )


@app.get("/history/{session_id}")
def get_history(session_id:str,limit:int = 10):

    message_list = history_repository.list_recent(session_id=session_id,limit=limit)

    logger.info(f"完成：{session_id}对应的历史记录查询！查询的数据数量：{len(message_list)}")

    return HistoryResponse(
        session_id=session_id,
        items=[
            HistoryItemResponse(
                id=str(message.get("_id")),
                session_id=message.get("session_id"),
                role=message.get("role"),
                text=message.get("text"),
                rewritten_query=message.get("rewritten_query"),
                item_names=message.get("item_names"),
                image_urls=message.get("image_urls"),
                ts=message.get("ts")
            )
            for message in message_list
        ]
    )


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=settings.app_host, port=settings.query_app_port)








