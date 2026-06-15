import json

from agents.mcp import MCPServerStreamableHttp

from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger, step_log
from app.infra.config.providers import infra_config
import asyncio

@step_log("get_rewritten_query_and_validate")
def get_rewritten_query_and_validate(state) -> str:

    # 1. 获取数据
    rewritten_query = state.get("rewritten_query")
    # 2. 校验
    if not rewritten_query:
        logger.error("rewritten_query没有内容,业务无法继续进行!")
        raise ValueError("rewritten_query没有内容,业务无法继续进行!")
    # 3. 返回
    return rewritten_query

@step_log("web_search_docs")
async def web_search_docs(rewritten_query:str):
    # -> CallToolResult
    """
    网络调用
    :param rewritten_query:
    :return:
    """
    # 1. 初始化mcp_server
    mcp_server = MCPServerStreamableHttp(
        name="web_search_mcp",
        client_session_timeout_seconds=300,
        params={
            "url": infra_config.mcp.mcp_base_url,
            "headers": {"Authorization": f"Bearer {infra_config.mcp.api_key}"},
            "timeout": 300
        },
        cache_tools_list=True,
        max_retry_attempts=3
    )
    try:
        # 2. 创建链接
        await mcp_server.connect()
        # 3. 调用网络工具
        tool_list = await mcp_server.list_tools()
        logger.info(f"本次链接服务对应的工具列表：{tool_list}")
        mcp_result = await mcp_server.call_tool(tool_name="bailian_web_search",arguments={"query":rewritten_query,"count":5})
        return mcp_result
    except Exception as e:
        logger.exception(f"调用工具出现问题,本次参数:{rewritten_query},错误原因:{str(e)}")
    finally:
        # 4. 断开链接
        await mcp_server.cleanup()


@step_log("search_by_web")
def search_by_web(state: QueryGraphState) -> QueryGraphState:
    """
    网络搜索服务：
    1. 通过 MCP 协议异步调用百炼联网搜索接口
    2. 将用户的查询转化为实时的、结构化的网络搜索结果
    3. 包含标题、链接和摘要
    4. 回写 web_search_docs
    """
    # 1. 获取和校验参数
    rewritten_query = get_rewritten_query_and_validate(state)
    # 2. 调用业务的网络搜索工具
    mcp_result = asyncio.run(web_search_docs(rewritten_query))
    logger.info(f"查询到的结果：{mcp_result}")
    # 3. 获取结果
    search_text = mcp_result.content[0].text    # 讨论为啥要这么取值
    # 转成dict / pages 对应的列表即可
    web_search_docs_list = json.loads(search_text).get("pages",[])
    logger.info(f"{rewritten_query}问题对应联网查询的结果：{web_search_docs_list}")
    return web_search_docs_list