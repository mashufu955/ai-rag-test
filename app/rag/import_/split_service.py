import json
import re
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import CHUNK_MAX_SIZE, CHUNK_SIZE, CHUNK_OVERLAP
from app.shared.runtime.logger import step_log, logger


@step_log("load_markdown_content")
def load_markdown_content(state: ImportGraphState) -> tuple[str, str, Path]:
    """
    从状态字典中安全加载 Markdown 内容和文档标题
    1. 优先从 state 中直接读取
    2. 缺失时自动从文件读取兜底
    3. 统一换行符格式，保证文本感觉
    :return: (处理后的md内容，文件标题)
    """
    # 从状态中获取核心数据：md内容、文件标题、md文件路径
    md_content = state.get("md_content")
    file_title = state.get("file_title")
    md_path = state.get("md_path")
    # ===================== 处理 md_content 缺失场景 =====================
    # 如果状态中没有md内容，尝试从本地md文件读取（兜底逻辑）
    if not md_content:
        logger.warning("没有从state读取到md_content内容，我们使用md_path尝试再次读取！")
        # 如果文件路径存在，则读取文件内容
        if md_path:
            md_content = Path(md_path).read_text(encoding="utf-8")
            state["md_content"] = md_content # 读取后回填到状态，避免重复读取
        # 双重校验：仍然无内容，直接抛出异常终止流程
        if not md_content:
            raise ValueError("md_content没数据，并且尝试读取md_path依然没有数据，终止执行！！")
    # ===================== 处理 file_title 缺失场景 =====================
    # 如果标题为空，使用文件名（无后缀）作为标题；无路径则使用默认值
    if not file_title:
        file_title = Path(md_path).stem if md_path else "default"
        state["file_title"] = file_title    # 回填到状态
    # ===================== 统一文本格式 =====================
    # 替换所有换行符为 \n，解决 Windows/Linux 换行符不一致问题
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    # 返回处理好的文本内容 + 标题，给后续切块使用
    return md_content, file_title, Path(md_path)

@step_log("split_by_titles")
def split_by_titles(md_content: str, file_title: str) -> list[dict]:
    """
    按 Markdown 标题（#、##、###...）进行【语义化文档切块】
    特点：
        1. 自动识别标题，保证段落语义完整
        2. 跳过代码块内部的内容，不把 ```内的内容误判为标题
        3. 每个块包含： 内容、当前标题、文档标题，方便后续检索
    :param md_content: Markdown 文本内容
    :param file_title: 文档名称（用于溯源）
    :return: 切块列表，每个元素是 {content, title, file_title}
    """
    # 正则：匹配 Markdown 标题（# ~ ###### 开头的行）
    reg = re.compile(r"^\s*#{1,6}\s.+")
    # 将全文按换行符切割成逐行处理
    lines = md_content.split("\n")
    # 存储最终切块结果
    chunks: list[dict] = []
    # 当前正则拼接的块标题
    current_title = None
    # 当前块的所有行内容
    current_title_lines: list[str] = []
    # 标记： 是否处于代码块（```...```）内部
    is_code_block = False
    # 记录切块数量
    chunk_size = 0
    # 逐行遍历 MD 内容
    for raw_line in lines:
        line = raw_line.strip()
        # 空行跳过
        if not line:
            logger.warning("处理行为空行，跳过本次筛选！")
            continue
        # ===================== 代码块判断 =====================
        # 遇到 ``` 或 ~~~ 标记，切换代码块状态
        if line.startswith("```") or line.startswith("~~~"):
            is_code_block = not is_code_block
            current_title_lines.append(line)  # 把代码行加入当前块
            continue
        # ===================== 识别标题并切分 =====================
        # 如果当前行是标题，并且**不在代码块内**，才进行切分
        if reg.match(line) and not is_code_block:
            # 如果已有上一个块内容，就把上一个块保存
            if current_title and len(current_title_lines) > 1:
                chunks.append({
                    "content": "\n".join(current_title_lines),  # 块内容
                    "title": current_title,  # 块标题
                    "file_title": file_title    # 文档名（溯源用）
                })
                chunk_size += 1
            # 以当前行作为新块的标题
            current_title = line
            current_title_lines = [current_title]
        # 普通行 -> 直接追加到当前块
        else:
            current_title_lines.append(line)
    # ===================== 保存最后一个块 =====================
    if current_title and len(current_title_lines) > 1:
        chunks.append({
            "content": "\n".join(current_title_lines),
            "title": current_title,
            "file_title": file_title
        })
        chunk_size += 1
    # ===================== 兜底：全文无标题时 =====================
    if chunk_size == 0:
        chunks.append({
            "content": md_content,
            "title": "default",
            "file_title": file_title
        })
    logger.info(f"完成文档语义切割，共计切出：{chunk_size}块！ 切块内容：{chunks}")
    return chunks

@step_log("_split_long_section")
def _split_long_section(section: dict[str, Any], max_length: int = CHUNK_MAX_SIZE) -> list[dict[str, Any]]:
    """
    内部工具函数：拆分【过长的文本块】，保证单个chunk不超过最大长度限制
    核心逻辑：
        1. 检查内容长度，不长则直接返回
        2. 标题单独保留，只拆分正文内容
        3. 使用语义化拆分器，按段落、句子拆分，保证语义完整
    :param section: 待拆分的切块（包含title、content等）
    :param max_length: 单个块最大字符长度
    :return: 拆分后的子块列表
    """
    # 获取块的正文内容
    content = section.get("content", "") or ""
    # 1. content的格式清理
    #   #title \n line \n line \n line
    title = section.get("title")
    body = content
    if content.startswith(title):
        body = content[len(title):].lstrip()    # 去掉多余的空格
    # 2. 定义每块的固定前缀 和 块的有效长度
    # prefix = title + \n
    prefix = title + "\n"
    available_length = max_length - len(prefix)

    # 3. 定义初始化递归字符拆分器（LangChain官方工具）
    # 按 段落->换行->句子->空格 优先级拆分，保证语义完整性
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=available_length,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？"]
    )
    sub_sections = []
    # 4. 遍历拆分后的正文片段，生成子块
    for index, chunk_text in enumerate(splitter.split_text(body), start=1):
        text = chunk_text.strip()
        # 跳过空内容
        if not text:
            continue
        # 拼接完整内容： 标题 + 拆分后的正文
        full_text = (prefix + text).strip()

        # 构造子块，保留溯源信息，添加分区编号
        sub_sections.append({
            "title": f"{title}-{index}" if title else f"chunk-{index}", # 子标题：原标题-序号
            "content": full_text,                                       # 完整内容
            "parent_title": title,                                      # 父标题（用于溯源）
            "part": index,                                              # 序号（同一章节下的第N部分）
            "file_title": section.get("file_title")                     # 文档原始标题
        })

    logger.info(f"已经完成{title}对应块进行短切! 切后块数为:{len(sub_sections)} , 数据预览: {sub_sections}")
    # 返回拆分完成的所有子块
    return sub_sections

@step_log("_merge_short_chunks")
def _merge_short_chunks(final_chunks:list[dict],max_length:int = CHUNK_MAX_SIZE,min_length:int=CHUNK_SIZE) -> list[dict]:
    # 同一个标题，小于600，进行合并，合并后不能大于1000
    # 1. 生命合并后的列表结果
    final_merge_chunks = []
    # 2. 记录第一个指针chunk的位置 <--- 后续
    start_chunk = None
    # 3. 循环处理后续的chunk进行合并处理
    for next_chunk in final_chunks:
        # 第一次
        # 4. start_chunk没有赋值，把第一个赋值
        if not start_chunk:
            start_chunk = next_chunk
            continue
        # 第二次之后
        # 5. start content是否小于600 and next 是不是同一个父标题
        is_lt_chunk_size = len(start_chunk.get("content")) < min_length
        is_same_parent_title = start_chunk.get("parent_title") and start_chunk.get("parent_title") == next_chunk.get("parent_title")
        if is_lt_chunk_size and is_same_parent_title:
            # 同一个父标题 start长度小于600
            # 6. 清理next的标题内容，再判断合并长度 标题\n内容
            next_content_to_title = next_chunk.get("content")[len(next_chunk.get("parent_title")) + 2 :]
            start_content = start_chunk.get("content")
            # 7. 长度校验
            merged_content = start_content + "\n" + next_content_to_title
            if len(merged_content) <= max_length:
                start_chunk['content'] = merged_content
                logger.info(f"父标题：{start_chunk['parent_title']}, start: {start_chunk['title']}, next: {next_chunk['title']} 完成合并！")
            else:
                final_merge_chunks.append(start_chunk)
                start_chunk = next_chunk
                continue
        else:
            final_merge_chunks.append(start_chunk)
            start_chunk = next_chunk
    # 循环执行完毕了
    if start_chunk is not None:
        final_merge_chunks.append(start_chunk)
    return final_merge_chunks

@step_log("refine_chunks")
def refine_chunks(chunks: list[dict], max_len: int = CHUNK_MAX_SIZE, min_len: int = CHUNK_SIZE) -> list[dict]:
    """
        进行精细切割，一共分为三步！ 长切 / 短合 / 补全属性
    :param chunks: 原始内容
    :param max_len: 触发长切参数
    :param min_len: 触发短合参数
    :return: 最终处理后的chunk
    """
    # 定义 接收最终结果
    final_chunks = []

    # 1. 循环判断是否需要长切
    for chunk in chunks:
        if len(chunk["content"]) > max_len:
            # 拆分过长的块，并加入结果列表
            final_chunks.extend(_split_long_section(chunk, max_len))
        else:
            final_chunks.append(chunk)
    # 2. 短合并
    final_merge_chunks = _merge_short_chunks(final_chunks)
    # 3. 优化属性存在
    for chunk in final_merge_chunks:
        if "parent_title" not in chunk:
            chunk["parent_title"] = chunk["title"]
        if "part" not in chunk:
            chunk["part"] = 1
    # 4. 返回处理后结果
    return final_merge_chunks

@step_log("backup_chunks_json")
def backup_chunks_json(final_chunks:list[dict], md_path_obj:Path):
    """
    数据备份 字典 -> 文件名.json
    """
    # 获取文件对象
    json_path_obj = md_path_obj.parent / f"{md_path_obj.stem}.json"
    # 写出内容即可 .json -> 字符串
    json_path_obj.write_text(json.dumps(final_chunks, indent=4, ensure_ascii=False), encoding="utf-8")
    logger.info(f"数据完成备份，备份的位置：{str(json_path_obj)}")


@step_log("split_document")
def split_document(state: ImportGraphState) -> ImportGraphState:
    """
    """
    """
    文档切块核心节点（RAG 最关键步骤）
    功能： 加载增强后的 Markdown 内容 -> 按标题智能切块 -> 优化块大小 -> 备份切块结果 -> 写入状态
    输出： 将分块后的文本列表存入 state，供后续向量化、入库使用
        """
    # 1. 从状态中加载【增强后的Markdown内容】和【文档标题】
    md_content, file_title, md_path_obj = load_markdown_content(state)
    # 2. 按 Markdown 标题（#、##、###）进行【智能语义切块】（保持段落完整性）
    chunks = split_by_titles(md_content, file_title)
    # 3. 精细切割（涉及长切和短切处理，返回最终处理的chunks）
    final_chunks = refine_chunks(chunks)
    # 4. 备份final_chunks内容
    backup_chunks_json(final_chunks, md_path_obj)
    # 5. 修改state状态 chunks
    state['chunks'] = final_chunks
    return state