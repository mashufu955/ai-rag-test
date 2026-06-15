from typing import Any

from pydantic import BaseModel

class QueryRequestParam(BaseModel):
    query: str
    session_id: str
    is_stream: bool=False

class QueryStreamResponse(BaseModel):
    message: str
    session_id: str

class QueryNotStreamResponse(BaseModel):
    message: str
    session_id: str
    answer: str
    done_list: list
    image_urls: list

# 清空历史记录响应的结构
class HistoryCleanResponse(BaseModel):
    message:str
    deleted_count:int


# 查询历史聊天记录的结构
class HistoryItemResponse(BaseModel):
    id:str
    session_id:str
    role:str
    text:str
    rewritten_query:str
    item_names:list
    image_urls:list | None
    ts:Any

class HistoryResponse(BaseModel):
    session_id:str
    items:list[HistoryItemResponse]