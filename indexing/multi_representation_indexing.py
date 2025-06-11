import uuid
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
   Indexing之Multi-representation Indexing 
    检索增强生成 (RAG) 系统中的一种高级索引技术，它通过为文档创建多种不同的表示形式来优化检索过程
    传统的 RAG 系统通常只使用单一向量表示文档，而多表示索引技术则创建多个向量表示，每个表示捕获文档的不同方面或特征

    
    1. 加载文档并分割
    2. 多视角表示生成: 生成基于内容的表示(捕获文本语义) | 生成基于结构的表示(捕获文档层次和关系) | 生成基于元数据的表示(捕获时间、作者等外部信息)
    3. 索引融合：开发策略将多个索引的检索结果进行合并和排序
    4. 检索优化：根据查询特性动态选择最相关的表示进行检索


    适用场景
        企业知识库检索：处理包含多种文档类型和格式的大型知识库
        长文档问答系统：如法律文书、学术论文或技术手册的问答
        多语言检索环境：需要同时处理语义和语言差异
        专业领域检索：如医疗、金融等需要精确和多维度信息的领域
        个性化搜索系统：根据用户偏好和历史行为调整检索策略
        复杂查询场景：需要结合多种信息源和上下文的复杂问题

"""


# 加载网页文档 - 第一个来源：关于AI代理的文章
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

# 加载第二个网页文档 - 关于人类数据质量的文章
loader = WebBaseLoader("https://lilianweng.github.io/posts/2024-02-05-human-data-quality/")
docs.extend(loader.load())

# 创建一个总结链：提取文档内容 -> 生成总结提示 -> 使用OpenAI模型 -> 解析输出为字符串
chain = (
    {"doc": lambda x: x.page_content}  # 提取文档的文本内容
    | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")  # 创建总结提示模板
    | ChatOpenAI(model="gpt-3.5-turbo",max_retries=0)  # 使用GPT-3.5模型生成总结
    | StrOutputParser()  # 将模型输出解析为字符串
)

# 批量处理所有文档，生成总结，最大并发数为5
summaries = chain.batch(docs, {"max_concurrency": 5})


# 创建向量存储，用于索引总结内容
vectorstore = Chroma(collection_name="summaries", embedding_function=OpenAIEmbeddings())

# 创建内存存储，用于存储原始文档
store = InMemoryByteStore()
id_key = "doc_id"  # 用于关联总结和原始文档的ID键名

# 创建多向量检索器
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,  # 存储总结的向量数据库
    byte_store=store,  # 存储原始文档的存储层
    id_key=id_key,  # 关联键
)

# 为每个文档生成唯一ID
doc_ids = [str(uuid.uuid4()) for _ in docs]

# 创建包含总结内容和关联ID的文档对象
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]

# 将总结文档添加到向量存储中
retriever.vectorstore.add_documents(summary_docs)
# 将原始文档存储到内存存储中，建立ID与文档的映射
retriever.docstore.mset(list(zip(doc_ids, docs)))

# 示例查询：关于"代理中的记忆"的信息
query = "Memory in agents"
# 使用向量存储直接检索相关总结文档（不使用多向量检索器）
sub_docs = vectorstore.similarity_search(query,k=1)
print(sub_docs[0])  # 打印检索到的总结文档

# 使用多向量检索器检索相关原始文档
retrieved_docs = retriever.get_relevant_documents(query, n_results=1)
# 打印检索到的原始文档的前500个字符
print(retrieved_docs[0].page_content[0:500])
