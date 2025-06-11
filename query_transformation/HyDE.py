from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


"""
    Query Transformations之HyDE(Hypothetical Document Embedding)
    先让 LLM 生成一个假设的回答，然后用这个回答作为检索的查询
    增强查询表示：原始问题可能比较简短或模糊，生成的假设文档提供了更丰富的语义表示，帮助检索系统找到更相关的内容
    语义扩展：LLM 生成的假设文档会包含与问题相关的更多概念、术语和上下文，扩大了检索的语义范围
    对齐用户意图：生成的假设文档更好地反映了用户问题的意图，减少了检索结果的偏差
    结构化查询：对于复杂问题，假设文档可能会包含更结构化的表示，帮助检索系统理解问题的各个方面

    1. 加载文档并分割
    2. 使用向量数据库存储分割文档chunk, 并使用向量数据库声明检索器
    3. 使用llm为用户的查询生成一个假设的回答, 并使用这个假设的回答作为检索的查询, 来检索到假设回答最为相关的文档chunk
    4. 使用假设的回答检索到的文档快chunk输入给llm, 让llm回答

    适用场景
        复杂问题解答:当用户问题复杂或涉及多个概念时, HyDE 可以帮助生成更全面的查询
        专业领域问答：在医学、法律、科学等专业领域，用户问题可能使用非专业术语，但 HyDE 可以生成更专业的查询表示
        模糊问题处理: 当用户问题表述模糊或不完整时, HyDE 可以帮助完善查询意图
        少样本学习场景: 在训练数据有限的情况下, HyDE 可以增强模型对问题的理解能力

"""

# 1. 加载文档并分割
# 1.1 网页文档加载
web_doc_loader = WebBaseLoader(web_path="https://lilianweng.github.io/posts/2023-06-23-agent/")
web_doc = web_doc_loader.load()

# 1.2 分割文档
doc_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=50
)
web_doc_chunk = doc_splitter.split_documents(documents=web_doc)


# 2. 使用向量数据库存储分割文档chunk, 并使用向量数据库声明检索器
# 2.1 使用向量数据库存储分割文档
vector_db = Chroma.from_documents(documents=web_doc_chunk, embedding=OpenAIEmbeddings())
# 2.2 声明检索器
retriever = vector_db.as_retriever()



# 3. 使用llm为用户的查询生成一个假设的回答, 并使用这个假设的回答作为检索的查询
# 3.1 为使用llm对用户的查询生成假设的回答定义提示词模板
hyde_templte = """Please write a scientific paper passage to answer the question
Question: {question}
Passage:
"""
hyde_prompt = ChatPromptTemplate.from_template(hyde_templte)

# 3.2 定义使用llm生成假设回答的chain
generate_hyde_query_chain = (
    hyde_prompt | ChatOpenAI(temperature=0) | StrOutputParser()
)

# 3.3 使用chain生成假设回答的chain
question = "What is task decomposition for LLM agents?"
hyde_query = generate_hyde_query_chain.invoke({"question": question})


# 3.4 将假设性回答输入给检索器, 检索出假设性回答最为相关的文档chunk
retrieval_doc_chain = generate_hyde_query_chain | retriever 
# 3.5 检索到假设的回答的最相关文档chunk
hyde_retrieval_doc = retrieval_doc_chain.invoke({"question": question})


# 4. 使用假设的回答检索到的文档快chunk输入给llm, 让llm回答
# 4.1 定义提示词模板
template = """Answer the following question based on this context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template=template)

final_hyde_chain = (
    prompt |
    ChatOpenAI(temperature=0) |
    StrOutputParser()
)

final_hyde_respose = final_hyde_chain.invoke({"context": hyde_retrieval_doc, "question": question})
print(final_hyde_respose)
