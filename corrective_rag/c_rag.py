from langchain import hub
from typing import List
from pprint import pprint
from langchain.schema import Document
from typing_extensions import TypedDict
from pydantic import BaseModel, Field  # 导入LangChain和Pydantic的基础模型类和字段装饰器
from langchain_community.vectorstores import Chroma
from langgraph.graph import END, StateGraph, START 
from langchain.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())


"""
    解释C-RAG(Corrective-RAG)在传统RAG的基础上引入了一个关键创新:  
        引入对检索到文档chunk的置信度评估, 即使用LLM对检索到的文档chunk进行评估, 评估文档chunk是否能够正确回答出用户的查询
        分为三个等级: Correct(正确)、Ambiguous(模糊不清)、Incorrect(不正确), 针对不同置信度采取不同的方案: 
            (1) 针对Correct和Ambiguous的文档chunk进行知识精炼和过滤, 得到更为准确和高质量的上下文信息(知识)
            (2) 针对Incorrect和Ambiguous的文档chunk所对应的用户查询进行改写, 使用改写后的用户查询使用联网搜索的方式,
                将检索到的内容与用户查询做匹配过滤, 得到更为准确和高质量的上下文信息(知识)

        最后, 将以上两个方案得出的高质量输出和原始用户查询一起输入给LLM, 让给LLM生成


        C-RAG的具体做法:



"""


# 1. 加载并分割文档
# 1.1 这里加载三个网页文档
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# 将多个文档列表合并为一个文档列表
docs_list = []
for url in urls:
    docs_list.extend(WebBaseLoader(url).load())

# 1.2 分割文档 
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250,
    chunk_overlap=0
)

# 分割出文档chunk
doc_splits = text_splitter.split_documents(docs_list)


# 存入向量数据库
vector_db = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings()
)

# 1.3 使用向量数据库定义召回器
retriever = vector_db.as_retriever()


# 2. 定义评估检索到的文档chunk是否与用户查询相关
# 2.1 定义用户结构化输出的类, 该类设定llm对文档评估是否相关后, 仅输出相关或不相关
class GradeDocments(BaseModel):
    """
        Binary score for relevance check on retrieved documents
        对检索到的文档进行相关性检查的二分类得分:
            如果相关输出yes, 否则输出no
    """
    binary_score:str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# 2.2 定义用于评估文档chunk是否与用户查询相关的llm
grade_score_evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# 将该llm定义好结构化输出
structured_llm_grader = grade_score_evaluator_llm.with_structured_output(GradeDocments)

# 2.3 定义用户评估chunk相关性的提示词
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),        # 将文档chunk和用户查询输入
    ]
)

# 2.4 定义好chunk相关性评估链chain
retriever_grader_chain = grade_prompt | structured_llm_grader

# 2.5 测试一下上面文档chunk相关性评估链
# 2.5.1 定义用户查询
chunk_relevant_question = "agent memory"
# 2.5.2 定义根据用户查询召回最相关的文档信息
# revevant_doc = retriever.get_relevant_documents(chunk_relevant_question)  # 老版本的调用方法
revevant_doc = retriever.invoke(chunk_relevant_question)    
# 2.5.3 输出一个最为相关的文档chunk看看
doc_text = revevant_doc[1].page_content
response = retriever_grader_chain.invoke({"question": chunk_relevant_question, "document": revevant_doc})
# print(response)

# 3. 定义用于生产的llm
# 3.1 定义提示模版和用于生成的llm
generate_prompt = hub.pull("rlm/rag-prompt")
generate_llm = ChatOpenAI(model="gpt-4", temperature=0)

# 3.2 定义输出后处理, 将输出使用\n\n连接在一起
def format_post_processing_docs(docs):
    return "\n\n".join(doc.paga_content for doc in docs)

# 3.3 定义用于生成的chain, 并使用结构化输出
generate_chain = generate_prompt | generate_llm | StrOutputParser()

# 3.4 尝试生成: 这里还是使用上面chunk_relevant_question
generate_reponse = generate_chain.invoke({"context": revevant_doc, "question": chunk_relevant_question})
# print(generate_reponse)


# 4. 定义用于重写用户查询的模块
# 当检索到的文档chunk没有一个与用户查询相关时, 或生成模块的llm生成的内容没有一个可用时, 需要启动查询重写模块, 将用户的查询进行重写

# 4.1 定义用于重写用户查询的llm
query_rewriter_llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# 4.2 定义用于重写用户查询的提示词
system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)
# 4.4 定义用于用户查询重写的chain
question_rewriter = re_write_prompt | query_rewriter_llm | StrOutputParser()
question_rewriter_response = question_rewriter.invoke({"question": chunk_relevant_question})
# print(question_rewriter_response)


# 5. 定义网络查询模块
web_search_tool = TavilySearchResults(k=3)      # 查询后取3个结果


"""
    以上5个模块是搭建C-RAG必须组件, 现在需要将5个组件搭建一个具有循环功能的结构
    让这个结构能够进行自主检索、自主选择何时应该进行联网搜索、何时应该进行查询重写

    因此, 使用LangGraph能够很方便搭建这样的选择、循环结构

        第一: 创建LangGraph中的Graph State, 即图的节点
        第二: 创建LangGraph中的Graph State, 即图的边, 边表示节点与节点之间的关系, 有选择关系, 循环关系等

"""

# 6. 定义表示Graph State 图状态的类
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str                   # 用户查询
    generation: str                 # llm针对用户查询和相关的文档chunk生成的回答
    documents: List[str]            # 检索到的文档chunk


# 7.1 定义检索操作节点
def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


# 7.2 定义生成节点
def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = generate_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

# 7.3 定义评估检索到的文档chunk节点, 如果结果为不相关, 则需要调用联网搜索
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retriever_grader_chain.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


# 7.4 定义用于用户查询重写的操作节点
def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


# 7.5 联网搜索
def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"documents": documents, "question": question}



# 8. 定义如上操作的关系边

# 8.1 定义什么时候进行生成、什么时候进行联网搜索的关系边
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    web_search = state["web_search"]
    state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"
    

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # 检索节点
workflow.add_node("grade_documents", grade_documents)  #  评估检索文档chunk的节点
workflow.add_node("generate", generate)  # 生成节点
workflow.add_node("transform_query", transform_query)  # 查询重写节点
workflow.add_node("web_search_node", web_search)  # 联网搜索节点

# 建立图结构
workflow.add_edge(START, "retrieve")                     # 检索节点设为开始节点
workflow.add_edge("retrieve", "grade_documents")         # 检索完后进行评估检索结果, 所以在检索节点和评估节点节点之间添加边
workflow.add_conditional_edges(                          # 添加条件边: 即评估检索到的chunk的结果是可行的, 则进行llm生成和生成的评估, 否则进行用户查询改写
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "web_search_node")  # 添加边: 查询改写后, 还是调用web联网检索
workflow.add_edge("web_search_node", "generate")         # 添加边: 联网检索后, 需要将检索的内容输入给llm进行生成
workflow.add_edge("generate", END)                       # 添加边: 生成节点与结束节点之间连接

# 编译工作流程
app = workflow.compile()

# 9. 测试C-RAG得效果
inputs = {"question": "Explain how the different types of agent memory work?"}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    pprint("\n---\n")

# Final generation
pprint(value["generation"])