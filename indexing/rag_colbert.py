import requests                                 # 导入requests库用于发送HTTP请求
from ragatouille import RAGPretrainedModel      # 加载ColBERTv2.0预训练的检索增强生成(RAG)模型
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain.storage import InMemoryByteStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

"""
   Indexing之colBert模型来做Indexing技术
    ColBERT (Contextualized Late Interaction over BERT) 是一种 上下文感知的后期交互 检索模型
    使用预训练模型ColBert将文档生成稠密嵌入向量, 它能表征丰富的语义信息
    加速查询响应：避免在查询时对整个文档库进行线性扫描
    提高检索质量：通过向量表示捕获语义信息
    优化资源利用：减少计算开销和内存占用


    适用场景


"""

# 临时修复 AdamW 导入问题
try:
    from transformers import AdamW
except ImportError:
    from torch.optim import AdamW
    import transformers
    transformers.AdamW = AdamW
    

# 加载ColBERTv2.0预训练的检索增强生成(RAG)模型
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")


def get_wikipedia_page(title: str):
    """
    从维基百科API获取指定标题页面的全文内容

    :param title: 维基百科页面标题
    :return: 页面的纯文本内容，如果不存在则返回None
    """
    # 维基百科API的端点URL
    URL = "https://en.wikipedia.org/w/api.php"

    # 配置API请求参数
    params = {
        "action": "query",      # API动作：查询页面
        "format": "json",       # 返回JSON格式数据
        "titles": title,        # 指定要查询的页面标题
        "prop": "extracts",     # 获取页面摘要内容
        "explaintext": True,    # 返回纯文本格式而非HTML
    }

    # 设置自定义User-Agent以符合维基百科API使用规范
    headers = {"User-Agent": "RAGatouille_tutorial/0.0.1 (ben@clavie.eu)"}

    # 发送GET请求并获取响应
    response = requests.get(URL, params=params, headers=headers)
    # 将响应内容解析为JSON格式
    data = response.json()

    # 从响应中提取页面内容
    # 注意：查询结果是字典形式，页面ID作为键，使用next(iter())获取第一个值
    page = next(iter(data["query"]["pages"].values()))
    # 返回页面的文本内容，如果不存在则返回None
    return page["extract"] if "extract" in page else None


# 获取"Hayao_Miyazaki"（宫崎骏）的维基百科页面全文
full_document = get_wikipedia_page("Hayao_Miyazaki")

# 使用RAG模型为文档创建索引
# - collection: 待索引的文档集合
# - index_name: 索引的唯一标识符
# - max_document_length: 每个文档片段的最大长度(词元)
# - split_documents: 是否将长文档分割成较小片段
RAG.index(
    collection=[full_document],
    index_name="Miyazaki-123",
    max_document_length=180,
    split_documents=True,
)

# 执行查询，返回与查询最相关的3个文档片段
results = RAG.search(query="What animation studio did Miyazaki found?", k=3)

# 将RAG模型转换为LangChain兼容的检索器
# k=3表示每次检索返回3个最相关的文档片段
retriever = RAG.as_langchain_retriever(k=3)

# 使用LangChain检索器执行相同的查询
# 该方法会返回结构化的检索结果，便于与LangChain生态集成
retriever.invoke("What animation studio did Miyazaki found?")