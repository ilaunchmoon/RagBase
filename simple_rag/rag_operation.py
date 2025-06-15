import bs4                                                                  # 常用的HTML/XML解析库，被许多网络爬虫和数据处理工具依赖, 需要pip install beautifulsoup4
from langchain import hub                                                   # 用于访问 LangChain Hub, 集中式的存储库, 用于存储和共享 LangChain 组件（如提示模板、链、代理等）, 用于下载langchain提供丰富的提示词模板和其他模板
from langchain.text_splitter import RecursiveCharacterTextSplitter          # 分割器
from langchain_community.document_loaders import WebBaseLoader              # web中html文件的加载器
from langchain_community.vectorstores import Chroma                         # 向量数据库
from langchain_core.output_parsers import StrOutputParser                   # 结构化输出解析, str数据解析
from langchain_core.runnables import RunnablePassthrough                    # 支持在链式调用中, 用于参数输入, 传递输入不做任何修改
from langchain_openai import ChatOpenAI, OpenAIEmbeddings                   # OpenAI模型的文档嵌入模型
from langchain_community.vectorstores import Chroma, FAISS, ElasticsearchStore


from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

# 加载文档
loader = WebBaseLoader(
    web_path="https://lilianweng.github.io/posts/2023-06-23-agent/"
)

doc = loader.load()         

# 文档分割
text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_spliter.split_documents(doc)


# 文档嵌入
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings()) 

retriever = vectorstore.as_retriever()          # 创建一个检索器，用于从向量数据库中查找与查询相关的文档块

# 找回和生成
# 1. 提示词
prompt = hub.pull("rlm/rag-prompt")             # 从langchain的hub下载rag专用的rag提示词模板

# 2. llm
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 3. 文档预处理
def format_doc(docs):
    return "\n\n".join(doc.page_content for doc in docs)        # 使用换行符 将分割后的文档 连接起来

# 4. 定义链式调用
"""
    首先通过 retriever 找到相关文档块，然后用 format_doc 格式化这些文档
    RunnablePassthrough() 直接传递用户问题
    将格式化后的文档（context）和用户问题（question）输入到提示词模板中
    提示词模板生成完整的提示文本，输入到语言模型中
    最后用 StrOutputParser() 将模型输出解析为字符串
"""
chain = (
    {"context": retriever | format_doc, "question": RunnablePassthrough()}      # 使用检索器检索到的内容和再将格式化的内容作为上下文信息content, 最后使用RunnablePassthrough() 直接传递用户问题
    | prompt    # 提示词
    | llm       # 大模型
    | StrOutputParser() # 字符串解析输出
)


# 查询问题
response = chain.invoke("这是关于什么文档, 100个字以内说清楚!")
print(response)





