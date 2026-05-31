from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.entry_service import resolve_input_file

"""
    @node_log("节点名") -> 方法进入 和 方法执行完毕 以及异常的日志
    节点方法（state : ImportGraphState）
        打印开始的日志
        1. add_running_task(state["task_id"], "node_entry")
        2. state = resolve_input_file(state) 调用节点业务函数 [节点 + 业务的关联]
        3. add_done_task(state["task_id"], "node_entry")
        开营结束的日志
        return state

    为啥add_running_task / add_done_task: 记录节点的完成状态，存储到对应task_id的列表中（此时：还是英文）
    为啥要传入task_id : 因为我们有很多客户端，为每个客户端存储一个对应的列表，最终存储到dict [task_id, []]
    节点名称是固定的么： 是的，因为后续 get 的时候使用列表推导式，将英文转成对应的中文，定义的时候 key 就是节点！
"""

@node_log("node_entry")
def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点（node_entry）
    为什么叫这个名字：作为图的 Entry Point, 负责接收外部输入并决定流程走向。
    """
    add_running_task(state["task_id"], "node_entry")
    state = resolve_input_file(state)
    add_done_task(state["task_id"], "node_entry")
    return state

if __name__ == '__main__':
    from app.shared.runtime.logger import logger
    from app.process.import_.agent.state import create_default_state
    import json

    # 单元测试：覆盖不支持类型、MD、PDF三种场景
    logger.info("===== 开始node_entry节点单元测试 =====")

    # 测试1: 不支持的TXT文件
    test_state1 = create_default_state(
        task_id="test_task_001",
        local_file_path="联想海豚用户手册.txt"
    )
    result_1 = node_entry(test_state1)
    print(f"第一次测试结果: \n {json.dumps(result_1, indent=4, ensure_ascii=False)}")
    # 测试2: MD文件
    test_state2 = create_default_state(
        task_id="test_task_002",
        local_file_path="小米用户手册.md"
    )
    result_2 = node_entry(test_state2)
    print(f"第二次测试结果: \n {json.dumps(result_2, indent=4, ensure_ascii=False)}")
    # # 测试3: PDF文件
    test_state3 = create_default_state(
        task_id="test_task_003",
        local_file_path="万用表的使用.pdf"
    )
    result_3 = node_entry(test_state3)

    print(f"第三次测试结果: \n {json.dumps(result_3, indent=4, ensure_ascii=False)}")

    # # 测试4: 没有传入local_file_path
    # test_state4 = create_default_state(
    #     task_id="test_task_004"
    # )
    # result_4 = node_entry(test_state4)
    #
    # print(f"第四次测试结果: \n {json.dumps(result_4, indent=4, ensure_ascii=False)}")


    logger.info("===== 结束node_entry节点单元测试 =====")