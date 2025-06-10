from langchain import hub
import tiktoken     # OpenAI 开发的专用分词工具, 在调用 API 前拆分长文本，确保符合模型输入要求, 防止输入文本超过模型最大 token 限制
import numpy as np
import bs4
from langchain import hub
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())


# 1. 演示如何使用嵌入模型, 输出嵌入之后的token数量
question = "What kinds of pets do I like?"
document = "My favorite pet is a cat. "

def num_tokens_from_string(string:str, encoding_name:str)->str:
    """
        返回文本字符串中的token数
    """
    encoding = tiktoken.get_encoding(encoding_name=encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# 比如使用词嵌入模型cl100k_base进行嵌入
print(int(num_tokens_from_string(question, "cl100k_base")))



# 2. 使用文本嵌入模型后, 结合相似度来检索处查询在文档中最为相关的文档片段
doc_embed = OpenAIEmbeddings()                  # 使用OpenAI的词嵌入模型, 必须使用OpenAI的API-Key
query_rest = doc_embed.embed_query(question)
doc_rest = doc_embed.embed_query(document)
# print(len(query_rest))

def cosine_similarity(vec1,  vec2):
    dot_product = np.dot(vec1, vec2)
    norm_v1 = np.linalg.norm(vec1)
    norm_v2 = np.linalg.norm(vec2)
    return dot_product / (norm_v1 * norm_v2)

# 将查询向量和文档向量进行计算相似度
similary = cosine_similarity(query_rest, doc_rest)      
print("Cosine Similarity: ", similary)




# 以下是一个完整RAG系统都具备的组件
# 1. 加载网页文档
loader = WebBaseLoader(
    web_path="https://lilianweng.github.io/posts/2023-06-23-agent/"
)

doc = loader.load()         

# 2. 分割文档
# 2.1 定义文档分割器
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=50
)

# 使用文档分割器, 进行分割
text_splitter_doc = text_splitter.split_documents(documents=doc)        # 执行分割文档操作

# 3. 使用向量数据库存放分割文档之后的文本chunk嵌入向量, 并使用向量数据库声明一个检索器
vector_db = Chroma.from_documents(
    documents=text_splitter_doc,
    embedding=OpenAIEmbeddings()
)

# 3.1 使用向量数据库声明一个检索器: 并且设置检索出与查询最为相关的1个文档内容
retriever = vector_db.as_retriever(search_kwargs={"k": 1})          # search_kwargs是字典类型

# relevant_doc = retriever.get_relevant_documents("What is Task Decomposition?")
relevant_doc = retriever.invoke("What is Task Decomposition?")      # 现在一般直接使用invoke()函数
# print(relevant_doc)


# 4. LLM生成: 将检索器检索到与查询最相似的内容和用户的查询一起输入到LLM, 让LLM输出回答
# 4.1 定义提示词模板 
query_prompt = """
Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(query_prompt)

# 4.2 定义llm
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 4.3 定义RAG chain
chain = prompt | llm 

chain_response = chain.invoke({"context": relevant_doc, "question": "What is Task Decomposition?"})
# print(chain_response)

# 4.4 一般RAG系统都会添加结构化解析输出, 如下解析加上使用LangChain给的RAG提示词模板和结构化解析输出
prompt_template_rag = hub.pull("rlm/rag-prompt")
# 注意由于{}是一个dict不支持runable, 即dict不是runable对象, 所以必须使用()框起来
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} |         # 注意: context的值, 直接传入检索器即可
    prompt_template_rag |
    llm |
    StrOutputParser()
)
rag_chain_response = rag_chain.invoke("What is Task Decomposition?")
print(rag_chain_response)


