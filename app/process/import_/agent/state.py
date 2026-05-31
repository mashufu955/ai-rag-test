# 定义主图全局的state
# 一共12个属性 state(TypeDict/BaseModel)
# 目标：1. 定义全局state 2. 提供便捷创建state的方法
import json
from typing import TypedDict
import copy
from app.shared.runtime.logger import logger

class ImportGraphState(TypedDict):
    # 最终任务状态
    task_id : str # 每次调用流程的标识
    # 文件状态判断
    is_md_read_enabled : bool
    is_pdf_read_enabled : bool
    # 地址路径内容
    local_file_path : str # 存储要解析文件的地址 pdf md
    local_dir : str  # 存储生成的md文件 pdf -> md
    md_path : str   # 专门存储md地址
    pdf_path : str  # 专门存储pdf地址
    file_title : str    # 存储文件名 没有后缀 ergouzi.pdf    ergouzi
    # 文本和切块内容
    md_content : str    # 读取md的内容，用于切片
    item_name : str     # 模型识别的一个文档对应的主体
    chunks : list       # 存储切块内容
    embeddings_content : list   # 存储带有向量的切块内容

# 提供下对外快速创建的方法
# 模板
default_state:ImportGraphState = {
    "task_id": "",
    "is_md_read_enabled": False,
    "is_pdf_read_enabled": False,
    "local_file_path": "",
    "local_dir": "",
    "md_path": "",
    "pdf_path": "",
    "file_title": "",
    "md_content": "",
    "item_name": "",
    "chunks": [],
    "embeddings_content": []
}

# 提供一个方法，可以返回我们state 并且可以根据传入参数进行对象的属性修改
#   1. 方法() -> default_state 2. 方法(参数) -> default_state（task_id = 传入参数） -> default_state
# 方法（task_id=007,local_file_path="./md.pdf"）
def create_default_state(**overriders) -> ImportGraphState:
    copy_state = copy.deepcopy(default_state)
    copy_state.update(overriders)
    return copy_state

def get_default_state() -> ImportGraphState:
    """
    返回一个新的状态实例，避免全局变量污染。
    """
    return copy.deepcopy(default_state)

if __name__ == '__main__':
    state = create_default_state(task_id="007")
    logger.info(f"测试复制方法： \n {json.dumps(state, ensure_ascii=False, indent=4)}")

    state1 = get_default_state()
    logger.info(f"测试复制方法： \n {json.dumps(state1, ensure_ascii=False, indent=4)}")