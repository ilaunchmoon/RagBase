from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads                         # 序列化和反序列化工具，用于将文档对象转换为字符串(Json字符)格式
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

"""
    演示使用Query-Transformations之RAG-Fusion重排融合的方式

    RAG-Fusion重排融合操作步骤:
        1. 读取文档并进行分割
        2. 使用向量数据库存放分割文档, 并生成检索器
        3. 使用LLM为用户查询生成多个类似查询
        4. 使用RRF算法对分割的文档chunk进行融合重排
        5. 使用chain结合融合重排后的文档信息, 输入LLM生成用户查询的答案  
    
    与multi-query的区别:  
        会发现RAG-Fusion和Multi-Query都是生成多个回答来进行查询转换

        维度	             Multi-Query	                                            RAG-Fusion
     目标优先级	          优先提升召回率，覆盖语义多样性        	                    优先提升准确率，解决结果冲突
     结果处理逻辑	      简单合并所有检索结果，依赖高频文档片段	                     复杂重排，通过 RRF 算法综合多查询排名，筛选高一致性文档
     对查询质量的依赖	   对查询变体的多样性要求高，需覆盖不同表述	                      对查询变体的相关性要求高，需确保各查询结果的一致性
     典型工具支持	      LangChain 的MultiQueryRetriever、SpaCy 的同义词扩展	      LangChain 的 RRF 算法集成、QDrant 的混合检索框架
     适用场景	         模糊查询、跨领域问题（如 “人工智能如何影响教育公平”）	           复杂分析、多维度验证场景（如 “加息对不同行业的影响”）

     总结:
        Multi-Query就有点像是Recall, 宁可错杀一千, 也不会放过一个
        RAG-Fusion就有点相似精确率, 不会冤枉一个
"""

# 1. 读取文档并进行分割
web_doc_loader = WebBaseLoader(web_path="https://lilianweng.github.io/posts/2023-06-23-agent/")
doc = web_doc_loader.load()

# 1.1 文档分割
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=50
)
# 使用文本分割器进行分割
doc_chunk = text_splitter.split_documents(documents=doc)


# 2. 使用向量数据库存放分割文档, 并生成检索器
# 2.1 使用向量数据库存放分割文档
vector_db = Chroma.from_documents(documents=doc, embedding=OpenAIEmbeddings())

# 2.2 使用向量数据库声明检索器
retriever = vector_db.as_retriever()


#  3. 使用LLM为用户查询生成4个类似查询
# 3.1 定义多查询提示词模版
prompts = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):
"""
rag_fusion_prompts = ChatPromptTemplate.from_template(prompts)

# 3.2 使用LLM来生成多个查询
generate_multi_querise = (
    rag_fusion_prompts |
    ChatOpenAI(temperature=0) | 
    StrOutputParser() |
    (lambda x: x.split("\n"))
)


#  4. 使用RRF算法对分割的文档chunk进行融合重排
# 4.1 定义用于RRF算法的重排融合函数
def reciprocal_rank_fusion(results: list[list], k=60):
    """
        接收多个排序列表 (每个列表代表一种检索策略的结果), 并通过 RRF 算法将它们融合成一个统一的排序结果
        参数 k 是 RRF 公式中的平滑因子，默认值为 60
    """

    # 创建一个空字典 fused_scores，用于存储每个文档的累计融合分数
    # 键是文档的字符串表示，值是对应的融合分数
    fused_scores = {}

    # 外层循环遍历每个检索结果列表（results 是一个列表的列表）
    for docs in results:
        # 内层循环遍历当前列表中的每个文档及其排名(使用 enumerate 获取索引作为排名, 从 0 开始)
        for rank, doc_item in enumerate(docs):
            # doc_str = dumps(doc)：将文档对象序列化为字符串，作为字典的键以确保唯一性
            doc_str = dumps(doc_item)
            # 如果文档尚未在fused_scores字典中，则将其添加为初始分数0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # 初始化或累加文档的融合分数：根据 RRF 公式 1 / (rank + k) 计算当前文档在当前列表中的得分，并累加到总分数中
            fused_scores[doc_str] += 1 / (rank + k)
    
    # 使用 sorted 函数对 fused_scores 字典按值（分数）降序排序
    # loads(doc): 将序列化的文档字符串反序列化为原始文档对象
    # 返回一个包含(文档对象, 融合分数)元组的列表，按分数从高到低排列
    reranked_results = [
        (loads(doc), score) for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # 返回一个包含 (文档对象, 融合分数) 元组的列表，按分数从高到低排列
    return reranked_results

# 4.2 生成多查询并使用重排融合算法RRF后的与多个子查询最相关的文档
question = "What is task decomposition for LLM agents?"
retrieval_chain_rag_fusion = generate_multi_querise | retriever.map() | reciprocal_rank_fusion
rag_fusion_docs = retrieval_chain_rag_fusion.invoke({"question": question})
print(len(rag_fusion_docs))



# 5. 使用chain结合融合重排后的文档信息, 输入LLM生成用户查询的答案  
# 5.1 定义提示词和提示词模版
query_template = """Answer the following question based on this context:
{context}
Question: {question}
"""
query_prompt = ChatPromptTemplate.from_template(query_template)

# 5.2 定义llm
llm = ChatOpenAI(temperature=0)

# 5.3 定义rag_fusion的chain
rag_fusion_chain = (
    {"question": RunnablePassthrough()} |  # 保留原始问题
    {"context": retrieval_chain_rag_fusion, "question": lambda x: x["question"]} |
    query_prompt |
    llm | 
    StrOutputParser()
)

# 5.4 定义rag_fusion_chain
rag_fusion_response = rag_fusion_chain.invoke({"question": question})
print(rag_fusion_response)







