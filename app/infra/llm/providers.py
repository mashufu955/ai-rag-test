from langchain_openai import ChatOpenAI

from app.infra.config.providers import infra_config
from app.shared.model import generate_embeddings, get_bge_m3_ef, get_llm_client, get_reranker_model

class LLMProvider:

    # 获取chat 文本模型  参数: 模型名字  JSON_Model
    def chat(self , model_name:str= None, json_mode:bool=False):
        return get_llm_client(model=model_name,json_mode=json_mode)

    # 获取vision_chat 视觉模型  允许传递 不传递给默认值
    def vision_chat(self,vision_model_name:str=None):
        model_name = vision_model_name or infra_config.llm.lv_model
        return get_llm_client(model=model_name)

    def embedding_mode(self):
        return get_bge_m3_ef()

    def embed_documents(self,documents:list[str]) -> dict[str,list]:
        """
            {
               dense: [[],[]], 1024
               sparse: [{index:xx},{}]
            }
        :param documents:
        :return:
        """
        return generate_embeddings(documents)

    def reranker_model(self):
        return get_reranker_model()
    
llm_provider  = LLMProvider()