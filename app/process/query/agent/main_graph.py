from langgraph.graph import StateGraph, END

from app.process.query.agent.nodes.node_answer_output import node_answer_output
from app.process.query.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.process.query.agent.nodes.node_rerank import node_rerank
from app.process.query.agent.nodes.node_rrf import node_rrf
from app.process.query.agent.nodes.node_search_embedding import node_search_embedding
from app.process.query.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.process.query.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger


# 1. 定义图构建对象
query_graph_builder = StateGraph(QueryGraphState)
# 2. 添加节点
query_graph_builder.add_node("node_item_name_confirm",node_item_name_confirm)
query_graph_builder.add_node("node_search_embedding",node_search_embedding)
query_graph_builder.add_node("node_search_embedding_hyde",node_search_embedding_hyde)
query_graph_builder.add_node("node_web_search_mcp",node_web_search_mcp)
query_graph_builder.add_node("node_rrf",node_rrf)
query_graph_builder.add_node("node_rerank",node_rerank)
query_graph_builder.add_node("node_answer_output",node_answer_output)

# 3. 添加起始节点和条件边
query_graph_builder.set_entry_point("node_item_name_confirm")
# answer判断 -> 有 -> 不确定item_name 没有识别到item_name
# 3.1 定义条件的路由函数
def node_item_name_confirm_after(state: QueryGraphState):
    if state.get("answer"):
        # 有值
        logger.info(f"本次没有明确的item_name,提前结束,待用户确定! {state.get('answer')}")
        return "node_answer_output"
    else:
        # 没有
        logger.info(f"有明确的item_names :{state.get('item_names')}业务继续进行即可!!")
        # 返回并发节点
        return "node_search_embedding","node_search_embedding_hyde","node_web_search_mcp"
# 3.2 设置条件边
query_graph_builder.add_conditional_edges(
    "node_item_name_confirm",
            node_item_name_confirm_after,
    {
        "node_answer_output":"node_answer_output",
        "node_search_embedding":"node_search_embedding",
        "node_search_embedding_hyde":"node_search_embedding_hyde",
        "node_web_search_mcp":"node_web_search_mcp"
    }
)
# 4. 添加静态边
query_graph_builder.add_edge("node_search_embedding","node_rrf")
query_graph_builder.add_edge("node_search_embedding_hyde","node_rrf")
query_graph_builder.add_edge("node_web_search_mcp","node_rrf")
query_graph_builder.add_edge("node_rrf","node_rerank")
query_graph_builder.add_edge("node_rerank","node_answer_output")
query_graph_builder.add_edge("node_answer_output",END)

# 5. 编译图对象
query_graph_app = query_graph_builder.compile()