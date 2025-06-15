from langchain import hub
from typing import List
from pprint import pprint
from typing_extensions import TypedDict
from pydantic import BaseModel, Field  # 导入LangChain和Pydantic的基础模型类和字段装饰器
from langchain_community.vectorstores import Chroma
from langgraph.graph import END, StateGraph, START 
from langchain.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())


"""
    解释Self-RAG(Self-Reflective RAG)的高级RAG的用法, 具备反思机制的RAG, 反思的点又如下三个:
        
        针对用户的输入是否需要触发向量数据库进行检索
        针对检索到的chunk判断它们是否与用户的查询相关
        针对检索到最为相关的chunk生成对应的答案判断这些生成的答案中是否具有支撑回答用户查询的内容

        让LLM自身具备“反思”和“决策”能力, 动态地决定在生成过程中何时、何地、以及如何检索外部知识
        这使得知识检索和生成过程不再是预定义或固定的步骤, 而是根据模型对自身生成内容质量的实时评估来灵活触发和控制
        从而显著提升了生成文本的质量、准确性、可信度和连贯性


        Self-RAG的具体做法:
            第一: 针对用户的输入使用LLM先判断该查询是否需要进行向量检索出最相似的多个文档chunk, 
                 如果不需要进行检索, 那么LLM会直接利用LLM本身就具备的知识库进行作答; 如果需要进行检索, 那么跳到第二步

            第二: 使用LLM对检索到的多个chunk进行评估, 评估对应的检索的文档chunk是否为相关的文档chunk
                 分为两个等级:  Relevant(相关)、Irrelevant(不相关), 直接丢弃不相关的, 如果都是不相关的, 则使用LLM改写用户(query transform)的查询返回到第一步, 重新检索判断

            第三: 针对上一步检索到的多个文档chunk输入LLM, LLM生成对应多个回答, 仔启动LLM对每个回答进行评估, 评估每个回答针对对应的查询是否为支持的回答
                 分为三个等级:  Supported(回答支撑查询)、Partially(回答部分支撑查询)、not Supported(回答无法支持查询), 直接丢弃后两种, 如果都是后两种, 再次使用LLM对用户查询改写, 重新返回到第一步

            第四: 针对前两步:  检索文档chunk和子查询的回答的标签进行排序, Relevant(相关) > Irrelevant(不相关); Supported(回答支撑查询) > Partially(回答部分支撑查询) > not Supported(回答无法支持查询)
                选择出排序最高(或top-k)的作为LLM的最后输入, 让LLM最终生成用户查询的答案, 将生成的答案再次使用LLM进行评估, 如果评估的结果是:  答案不可用, 需要再次使用LLM对用户的查询进行改写, 再次回到第一步, 重新开始

"""


# 1. 加载并分割文档
# 1.1 这里加载三个网页文档
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# 将多个文档列表合并为一个文档列表
web_doc_list = []
for url in urls:
    web_doc_list.extend(WebBaseLoader(url).load())

# 1.2 分割文档 
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250,
    chunk_overlap=0
)

# 分割出文档chunk
web_docs_splitter = text_splitter.split_documents(web_doc_list)


# 存入向量数据库
vector_db = Chroma.from_documents(
    documents=web_docs_splitter,
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
grade_score_evaluator_llm = ChatOpenAI(model="gpt-4", temperature=0)
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


# 4. 定义用户评估llm回答是否可用的模块, 即评估llm的回答是否可用
# 4.1 定义用户结构化输出的类, 该类设定llm对生成的回答评估是否可用, 仅输出可用或不可用, 也就是评估用于回答用户查询的llm是否发生幻觉
class GradeHallucinations(BaseModel):
    """
        Binary score for hallucination present in generation answer.
        用于评估llm的回答是否出现幻觉, 如果是则输出yes, 否则输出no
    """
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

# 4.2 定义用于评估回答是否有用的llm
grade_generator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_grade_generator = grade_generator_llm.with_structured_output(GradeHallucinations)

# 4.3 定义用于评估回答是否有用的提示词
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

# 4.4 定义评估链
hallucination_grader_chain = hallucination_prompt | structured_llm_grader
hallucination_response = hallucination_grader_chain.invoke({"documents": revevant_doc, "generation": generate_reponse})
# print(hallucination_response)       # 预计输出yes


# 5. 定义用于重写用户查询的模块
# 当检索到的文档chunk没有一个与用户查询相关时, 或生成模块的llm生成的内容没有一个可用时, 需要启动查询重写模块, 将用户的查询进行重写

# 5.1 定义用于重写用户查询的llm
query_rewriter_llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# 5.2 定义用于重写用户查询的提示词
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
# 5.4 定义用于用户查询重写的chain
question_rewriter = re_write_prompt | query_rewriter_llm | StrOutputParser()
question_rewriter_response = question_rewriter.invoke({"question": chunk_relevant_question})
# print(question_rewriter_response)


"""
    以上5个模块是搭建Self-RAG必须组件, 现在需要将5个组件搭建一个具有循环功能的结构
    让这个结构能够进行自主反思、自主选择何时应该进行检索、何时应该进行查询重写

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


# 7. 定义操作模型的图节点: 检索操作节点、生成操作节点、评估检索到的文档是否是与用户查询相关的操作节点、评估生成的答案是否可用的操作节点

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


# 7.3 定义用于评估检索到的文档是否与用户查询相关的操作节点
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
            continue
    return {"documents": filtered_docs, "question": question}


# 7.4 评估生成的答案是否可用的操作节点



# 7.5 定义用于用户查询重写的操作节点
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




# 8. 定义如上几个操作之间的关系边

# 8.1 定义什么时候进行生成的关系边
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
    filtered_documents = state["documents"]

    if not filtered_documents:
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
    
# 8.2 定义生成评估和回答评估之间的关系边
def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # 评估llm的回答是否发生幻觉, 是否可用
    score = hallucination_grader_chain.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = grade_generator_llm.invoke({"question": question, "generation": generation})            # 对llm的生成进行评估
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    

# 9. 构建Graph图
workflow = StateGraph(GraphState)

# 9.1 定义节点
workflow.add_node("retrieve", retrieve)  # 检索节点
workflow.add_node("grade_documents", grade_documents)  # 评估检索文档chunk的节点
workflow.add_node("generate", generate)  # 生成节点
workflow.add_node("transform_query", transform_query)  # 用户查询重写节点

# Build graph
workflow.add_edge(START, "retrieve")                   # 检索节点设为开始节点
workflow.add_edge("retrieve", "grade_documents")       # 检索完后进行评估检索结果, 所以在检索节点和评估节点节点之间添加边
workflow.add_conditional_edges(                        # 添加条件边: 即评估检索到的chunk的结果是可行的, 则进行llm生成和生成的评估, 否则进行用户查询改写
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")    # 添加边: 查询改写后, 还是需要返回到检索这一步, 则在检索操作节点 和 查询改写节点之间添加边
workflow.add_conditional_edges(                     # 添加条件边: 生成节点和生成评估节点之间添加条件边, 如果生成评估结果是可用, 则结束; 如果是不可支撑, 则重新生成; 如果是不可用, 则进行查询重写操作
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

# 编译工作流
app = workflow.compile()


# 9. 测试Self-RAG得效果

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

