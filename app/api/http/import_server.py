"""
导入服务 HTTP 入口模块，直接承载导入接口与相关接口业务逻辑。
"""
import shutil
import sys
import uuid
from datetime import datetime
from mimetypes import guess_type
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware

from app.api.schema.import_schema import TaskStatusSchema, UploadSchema
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.process.import_.agent.main_graph import kb_import_app
from app.process.import_.agent.state import get_default_state, ImportGraphState, create_default_state
from app.infra.config.providers import settings
from app.shared.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    get_done_task_list,
    get_running_task_list,
    get_task_status,
    update_task_status, add_running_task, add_done_task
)

app = FastAPI(
    title=settings.import_app_name,
    description="企业化 RAG 导入服务，负责文件上传、导入执行状态与状态查询。",
    version="0.2.0"
)

# 跨域问题 CORS
# 后端 -> 响应头
app.add_middleware(
    CORSMiddleware,
    # 主机：端口
    allow_origins=list(settings.cors_origins) or ["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# 1. 返回import.html文件
@app.get("/html")
def html():
    html_path_obj = PROJECT_ROOT / "app" / "resources" / "html" / "import.html"
    return FileResponse(
        path=html_path_obj,
        media_type=guess_type(html_path_obj.name)[0]
    )


# 2. 返回task_id对应的任务状态
@app.get("/status/{task_id}")
def task_status(task_id:str):
    logger.info(f"获取任务状态接口被调用，task_id：{task_id}")
    return TaskStatusSchema(
        code=200,
        task_id=task_id,
        status=get_task_status(task_id),
        done_list=get_done_task_list(task_id),
        running_list=get_running_task_list(task_id)
    )

def invoke_graph(task_id:str, local_file_path:Path, local_dir:Path):
    """
        调用图对象 .invoke
    """
    state = create_default_state(task_id=task_id, local_file_path=str(local_file_path), local_dir=str(local_dir))

    try:
        logger.info(f"{task_id}对应的文件解析任务开始执行！参数state:{state}")
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        final_state = kb_import_app.invoke(state)
        logger.info(f"{task_id}对应的文件解析任务完成！最终结果为：{final_state}")
        update_task_status(task_id, TASK_STATUS_COMPLETED)
    except Exception as e:
        update_task_status(task_id, TASK_STATUS_FAILED)
        logger.exception(f"===== 全流程测试运行失败 =====")

# 3. 上传文件的异步接口  post / upload files : 文件列表  后天执行图过程
@app.post("/upload")
def upload_and_invoke_graph(backgroundtasks:BackgroundTasks, files:list[UploadFile]):

    """
        1. 接收上传的文件（文件存储到项目下）
        2. 异步执行导入图对象（state local_file_path, local_dir, task_id） 10 20s
        3. 直接返回结果
    :param backgroundtasks:
    :param files:
    :return:
    """
    # 约定存储的位置 /output / 时间 / task_id -> local_dir   +   文件名.pdf -> local_file_path
    # 1.1 接收上传的文件（文件存储到项目下）
    # 准备一个存储文件夹 [没有task_id]

    task_id = str(uuid.uuid4()) # 永远不重复的随机字符串
    add_running_task(task_id, "upload_file")
    local_dir_path_obj = PROJECT_ROOT / "output" / datetime.now().strftime("%Y%m%d") / task_id
    # 没有就建一个
    local_dir_path_obj.mkdir(parents=True, exist_ok=True)

    # 1.2 将文件存储到文件夹中
    current_file = files[0]
    local_file_path_obj = local_dir_path_obj / current_file.filename

    with local_file_path_obj.open("wb") as file_buffer:
        # copyfileobj 好处：
        # current_file [upload包装对象].file [上传文件] -> 写到file_buffer本地文件中
        # 循环读取 每次读取 64kb
        shutil.copyfileobj(current_file.file, file_buffer)

    add_done_task(task_id, "upload_file")

    # 2. 异步调用图，让图解析 local_file_path_obj local_dir_path_obj task_id
    backgroundtasks.add_task(
        invoke_graph, # 异步执行的函数名
        task_id=task_id,
        local_file_path=local_file_path_obj,
        local_dir=local_dir_path_obj
    )

    # 3. 返回结果
    return UploadSchema(
        code=200,
        message="文件上传成功！",
        task_ids=[task_id]
    )



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=settings.app_host, port=settings.import_app_port)