from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger,step_log
from pathlib import Path

@step_log("resolve_input_file")
def resolve_input_file(state: ImportGraphState) -> ImportGraphState:
    # 1. 获取state local_file_path 属性     state['key'] -> state.get(key)
    local_file_path = state.get("local_file_path")
    # 2. 进行local_file_path验证，为空 -> 打印异常日志，抛出异常
    if not local_file_path:
        logger.error("传入的local_file_path参数为空,没有文件,无法继续业务! 直接抛出异常!")
        raise ValueError("传入的local_file_path参数为空,没有文件,无法继续业务! 直接抛出异常!")
    # 3. 判断是md  is_md_read_enabled = True md_path = local_file_path
    if local_file_path.endswith(".md"):
        state['is_md_read_enabled'] = True
        state['md_path'] = local_file_path
    # 4. 判断是pdf  is_pdf_read_enabled = True pdf_path = local_file_path
    elif local_file_path.endswith(".pdf"):
        state['is_pdf_read_enabled'] = True
        state['pdf_path'] = local_file_path
    # 5. else啥也不是
    else:
        logger.warning(f"传入的文件:{local_file_path}类型无法处理,当前项目只支持 md / pdf类型,直接跳转到END节点!")
        return state
    # 6. 处理local_file_path -> file_title -> state
    state["file_title"] = Path(local_file_path).stem
    # 7. 返回state 处理完毕
    return state