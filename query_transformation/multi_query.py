import bs4
from langchain import hub    
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain.load import dumps, loads 

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

"""
    演示使用Query-Transformations之Multi-Query多查询的方式

    Multi-Query多查询操作步骤:
        1. 读取文档并进行分割
        2. 使用向量数据库存放分割文档, 并生成检索器
        3. 使用LLM为用户查询生成多个类似查询
        4. 生成多个查询后, 不同的子查询有可能会返回相同检索文档, 所以需去掉重复的检索文档
        5. 使用多查询检索到的最为相似的文档信息来让LLM生成查询的回答
"""


# 1. 网页文档加载
# 网页文档加载器
web_doc_loader = WebBaseLoader(web_path="https://lilianweng.github.io/posts/2023-06-23-agent/")
# 加载网页文档
doc = web_doc_loader.load()


# 2. 文档分割
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=50
)

# 使用分割器分割
doc_chunk = text_splitter.split_documents(documents=doc)

# 3. 使用向量数据库结合文本嵌入模型对文档chunk进行嵌入并存放, 最后使用向量数据库生成检索器
vector_db = Chroma.from_documents(documents=doc_chunk, embedding=OpenAIEmbeddings())
retriever = vector_db.as_retriever()        # 声明检索器


# 4. 定义多查询的提示词模板
prompt = """你是一名AI语言模型助手。你的任务是生成5个不同版本的用户问题，以从向量数据库中检索相关文档。
通过对用户问题生成多个视角，你的目标是帮助用户克服基于距离的相似性搜索的一些限制。提供这些用换行符分隔的替代问题。
原始问题：{question}
"""
prompt_template = ChatPromptTemplate.from_template(prompt)

# 5. 利用llm来生成多个用户查询的版本
generate_multi_query = (
    prompt_template | 
    ChatOpenAI(model="gpt-4", temperature=0) |
    StrOutputParser() |             # 使用str结构化输出解析
    (lambda x: x.split("\n"))       # 使用换行符分割llm生成的多查询
)

# print(generate_multi_query.invoke("What is task decomposition for LLM agents"))


# 6. 生成多个查询后, 不同的子查询有可能会返回相同检索文档, 所以需去掉重复的检索文档
def get_unique_retrievaled_doc(documents:list[list]):
    """ 
        用于返回检索到文档, 保证返回的是唯一, 不存在重复的检索文档
    """
    # dumps 将 LangChain 的 Document 对象序列化为字符串
    # loads 将字符串反序列化为 Document 对象

    # 扁平化嵌套列表，并将每个文档对象转为字符串
    flattend_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # 使用集合去重（集合不允许重复元素）
    unqiue_doc = list(set(flattend_docs))
    # 将字符串转回文档对象并返回
    return [loads(doc) for doc in unqiue_doc]

question = "What is task decomposition for LLM agents?"
retrieval_chain = generate_multi_query | retriever.map() | get_unique_retrievaled_doc
unique_retrieval_docs = retrieval_chain.invoke({"question": question})
# print(len(unique_retrieval_docs))


# 7. 使用多查询检索到的最为相似的文档信息来让LLM生成查询的回答
multi_query_prompt_template = """Answer the following question based on this context:

{context}

Question: {question}

"""

multi_query_prompt = ChatPromptTemplate.from_template(multi_query_prompt_template)
llm = ChatOpenAI(temperature=0)

multi_query_rag_chain = (
    RunnablePassthrough.assign(                     # 为了使用链式调用, 必须保证所有都是可runable的, 则需要使用RunnablePassthrough()
        context=lambda x: unique_retrieval_docs,
        question=lambda x: x["question"]
    ) |
    multi_query_prompt | 
    llm |
    StrOutputParser()
)

reponse = multi_query_rag_chain.invoke({"question": question})
print(reponse)
