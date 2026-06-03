# pdf_parse_service
# MinerU 模型版本配置（vlm = 视觉语言模型，适合PDF/图片高精度解析）
MINERU_MODEL_VERSION = "vlm"
# MinerU 任务轮询最大超时时间（单位：秒），超过则判定任务失败
# 600 -> 一个pdf 约等于 1秒
MINERU_POLL_TIMEOUT_SECONDS = 600
# MinerU 任务轮询间隔时间（单位：秒），每隔多久查询一次任务状态
MINERU_POLL_INTERVAL_SECONDS = 3
# MinerU 文件下载超时时间（单位：秒），下载文件超过此时长则中断
MINERU_DOWNLOAD_TIMEOUT_SECONDS = 30

CHUNK_MAX_SIZE = 1000
# 文本切块基准长度：单个文本块理想大小为 600 字符（兼顾语义完整性 + 检索精度）
CHUNK_SIZE = 600
# 文本块重叠长度：相邻块之间重叠 20 字符，保证语义不被切断、上下文连贯
CHUNK_OVERLAP = 50