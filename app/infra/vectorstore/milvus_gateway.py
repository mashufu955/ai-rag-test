
from app.infra.config.providers import infra_config
from app.shared.clients.milvus_utils import get_milvus_client

class MilvusGateway:

    @property
    def item_collection_name(self):
        return infra_config.milvus.item_name_collection
    @property
    def chunk_collection_name(self):
        return infra_config.milvus.chunks_collection
    @property
    def client(self):
        return get_milvus_client()

milvus_gateway = MilvusGateway()