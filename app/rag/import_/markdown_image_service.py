import base64
import re
from pathlib import Path
from typing import List, Dict
import mimetypes

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from minio.deleteobjects import DeleteObject

from app.infra.llm.providers import llm_provider
from app.infra.object_storage import minio_gateway
from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log
from app.shared.utils.rate_limit_utils import apply_api_rate_limit


@step_log("load_markdown_and_image_dir")
def load_markdown_and_image_dir(state) -> tuple[str, Path, Path]:
    # 1. 获取参数 md_content md_path
    md_path = state.get("md_path")
    md_content = state.get("md_content")
    # 2. md_path 非空校验
    if not md_path:
        logger.error("md_path为空，无法获取图片地址等，业务无法继续！")
        raise ValueError("md_path为空，无法获取图片地址等，业务无法继续！")
    # 3. md_content 进行非空校验 / 空给予默认值
    md_path_obj: Path = Path(md_path)
    if not md_content:
        logger.info("md_content为空，可能从md数据格式过来，根据md_path二次读取即可")
        md_content = md_path_obj.read_text(encoding="utf-8")
        if not md_content:
            logger.error(f"从{md_path}读取md_content内容失败，业务无法继续进行！")
            raise ValueError(f"从{md_path}读取md_content内容失败，业务无法继续进行！")
        # state 没有md_content，但是我们重新读取到了md_content
        state["md_content"] = md_content
    # 4. images对应Path获取
    images_path_obj = md_path_obj.parent / "images"
    # 5. 返回结果
    return md_content, md_path_obj, images_path_obj

SUPPORTED_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"]

def scan_images(md_content: str, image_path_obj: Path, context_length: int = 100) -> list[tuple[str,str,tuple[str,str]]]:
    images_context = []
    # 1. 从image_path_obj中获取每一个文件
    for image_file_obj in image_path_obj.iterdir():
        image_name = image_file_obj.name
        # 判断是不是图片
        if not image_file_obj.suffix in SUPPORTED_IMAGE_EXTENSIONS:
            # 不是图片
            logger.warning(f"{image_file_obj}不是图片，请检查！")
            continue
        # 2. 定义这张图片专属的正则规则
        reg = re.compile(r"\!\[.*?\]\(.*?" + re.escape(image_name)+ r".*?\)")
        match = reg.search(md_content)

        # 3. match校验，不存在，是图片，但是没有引用
        if not match:
            logger.warning(f"图片：{image_name}没有被md内容引用！无需处理，跳过本次循环！")
            continue

        # 4. match中的定位获取上下文数据
        start, end = match.span()   # match . start() end()
        pre_context = md_content[max(start-context_length, 0): start] # start-context_length<0 ->0
        post_context = md_content[end: min(end+context_length, len(md_content))] # end_context > len(max) -> len(max)
        images_context.append(
            (
                image_name,
                str(image_file_obj),
                (
                    pre_context,
                    post_context
                )
            )
        )
    logger.info(f"完成图片的上下文提取：{images_context}")
    return images_context

@step_log("summarize_images")
def summarize_images(image_context_list: list[tuple[str,str,tuple[str,str]]], stem: str) -> Dict[str, str]:
    """
    进行图片意图识别
    :param images_context_list: 图片名 地址 以及上下文
    :param stem: 图片所在的文件夹
    :return: {图片和对应的含义}
    """
    # 1. 获取视觉模型对象 llm/providers vision_chat()
    # 注意：修改 LLMProvider添加实例化 llm_provider = LLMProvider()
    vision_model = llm_provider.vision_chat()
    # 2. 准备一个存储含义的字典 images_dict [str, str] = {}
    images_summary_dict: Dict[str, str] = {}
    # 3. 循环 -> (图片名，地址，（上，下）) in    [(图片名，地址，（上，下）)]
    for image_name, image_path, (pre_context, post_context) in image_context_list:
        # 添加访问限制
        apply_api_rate_limit()
        # 4. 加载提示词和封装提示词
        image_summary_prompt = load_prompt("image_summary", root_folder=stem, image_content=(pre_context,post_context))
        image_path_obj = Path(image_path)
        image_base_str = base64.b64encode(image_path_obj.read_bytes()).decode(encoding="utf-8")
        # https://help.aliyun.com/zh/model-studio/vision#bc4fd98b485d
        human_message = HumanMessage(
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mimetypes.guess_type(image_name)[0]}; base64,{image_base_str}"}
                },
                {"type": "text", "text": f"{image_summary_prompt}"}
            ]
        )
        # 5. 和视觉模型进行交互
        vision_chains = vision_model | StrOutputParser()
        image_summary = vision_chains.invoke([human_message])
        # 6. 存储到对应字典
        images_summary_dict[image_name] = image_summary

    logger.info(f"完成图片内容识别，识别结果为：{images_summary_dict}")
    return images_summary_dict

@step_log("upload_images_and_replace")
def upload_images_and_replace(image_context_list:list[tuple[str, str, tuple[str, str]]],
                              image_summaries_dict: Dict[str, str], md_content: str, stem: str) -> str:
    """
        进行minio的文件上传和md_content内容替换
    :param images_context_list: [(图片名，图片地址，（上，下）)]
    :param images_summary_dict: {图片名：描述}
    :param md_content: md内容 ！[](./)
    :param stem: 烫金机
    :return: 新的md_content md内容 ！[描述](http...)
    """
    # 1. 删除原文件再minio中存储的图片信息
    """
        存储图片的路径 object_name
            image_dir -> 所有图片的公共前缀
                stem -> 对应每个文件的文件夹 方便进行文件的删除和查看
                    image_name.jpg -> 具体的图片
    """
    # 1.1 查询要删除的对象列表
    # todo minio的gateway 实例化一个对象
    # object_name
    list_object = minio_gateway.client().list_objects(
        bucket_name=minio_gateway.bucket_name,
        # 查询不到 前面多了 / [1:]
        prefix=f"{minio_gateway.image_dir[1:]}/{stem}", # 删除指定文件夹对应的图片
        recursive=True
    )
    delete_object_list = [DeleteObject(lo.object_name) for lo in list_object]
    # 1.2 根据对象列表进行删除
    errors = minio_gateway.client().remove_objects(
        bucket_name=minio_gateway.bucket_name,
        delete_object_list=delete_object_list
    )

    for error in errors:
        logger.warning(f"删除文件出现异常！{error}")
    logger.info("已经删除文件了！！")

    # 2. 循环传递每一张图片到minio的服务器
    image_minio_url_dict: Dict[str, str] = {}
    for image_name, image_path_str, _ in image_context_list:
        try:
            object_name = f"{minio_gateway.image_dir}/{stem}/{image_name}"
            minio_gateway.client().fput_object(
                bucket_name=minio_gateway.bucket_name,
                object_name=object_name,
                file_path=image_path_str,
                content_type=mimetypes.guess_type(image_name)[0]
            )
            # 3. 存储每张图片对应的minio的网络地址
            image_minio_url_dict[image_name] = minio_gateway.build_image_url(stem, image_name)
        except Exception as e:
            logger.warning(f"上传图片出现异常！{e}")

    # 4. 循环处理每一张图片，替换md_content内容
    for image_name, image_ur in image_minio_url_dict.items():
        image_summary = image_summaries_dict[image_name]

        reg = re.compile(r"\!\[.*?\]\(.*?" +re.escape(image_name)+ r".*?\)")
        md_content = reg.sub(lambda _ : f"![{image_summary}]({image_ur})", md_content)

    # 5. 返回新的md_content
    return md_content

@step_log("back_up_new_md_content")
def back_up_new_md_content(md_content_new, md_path_obj):
    # 新的地址 Path
    new_md_path_obj = md_path_obj.with_name(f"{md_path_obj.stem}_new.md")
    # 写出数据即可
    new_md_path_obj.write_text(md_content_new, encoding="utf-8")
    return str(new_md_path_obj)

@step_log("enrich_markdown_images")
def enrich_markdown_images(state: ImportGraphState) -> ImportGraphState:
    """
    Markdown 图片增强服务：
    1. 扫描 Markdown 中的图片
    2. 调用多模态模型生成图片说明
    3. 上传图片到 MinIO
    4. 替换 Markdown 图片地址并回写 md_content
    """
    # 1. 获取操作参数md
    md_content, md_path_obj, image_path_obj = load_markdown_and_image_dir(state)
    # 2. 判断image_path_obj是否存在内容，没有直接结束进行下一节点
    if not any(image_path_obj.iterdir()):
        # 空文件夹
        logger.warning(f"{image_path_obj}没有图片，无需图片处理！正常进入下一个节点！")
        return state
    # 3. 识别md_content图片的上下文
    # List[tuple[str, str, tuple[str, str]]] [图片名.jpg]
    images_context : List[tuple[str, str,tuple[str, str]]] = scan_images(md_content, image_path_obj)
    # 4. 使用视觉模型对图形进行意图识别
    # [图片的.png：描述]
    images_summary_dict = summarize_images(images_context, md_path_obj.stem)
    # 5. 上传图片并且替换md_content
    md_content_new = upload_images_and_replace(images_context, images_summary_dict, md_content, md_path_obj.stem)
    # 6. 备份新的md_content_new -> md_path_obj 烫金机.md 烫金机_new.md
    new_md_path_str = back_up_new_md_content(md_content_new, md_path_obj)
    # 7. 更新 state md_content md_path
    state['md_content'] = md_content_new
    state['md_path'] = new_md_path_str
    # 8. 返回结果
    return state