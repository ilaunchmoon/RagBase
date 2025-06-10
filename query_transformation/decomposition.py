from langchain import hub
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter

from dotenv import find_dotenv, load_dotenv
_ = load_dotenv(find_dotenv())


"""
    演示使用Query-Transformations之decomposition分解的方式, 即将复杂问题拆分为多个可独立搜索的子问题, 最后综合答案生成最终回答

    decomposition操作步骤:
        1. 读取文档并进行分割
        2. 使用向量数据库存放分割文档, 并生成检索器
        3. 使用LLM为用户查询分解为多个子问题(子查询)
        4. 对每个子问题分别进行文档检索
        5. 使用 RAG 技术为每个子问题生成回答, 对子问题的回答有两种方式--
            Answer recursively(递归回答):
                链式依赖：每个子问题的回答依赖于前一个子问题的答案
                逐步深入：通过迭代生成答案，后一步骤利用前一步骤的结果
                关键机制：每个子问题的回答会被添加到 q_a_pairs 中，后续子问题可以利用这些历史答案作为上下文
                适用场景：
                    -- 推理链问题：如数学证明、逻辑推理，每个步骤依赖前一步骤的结论
                    -- 复杂分析：如市场趋势分析，需要先分析历史数据，再预测未来
                    -- 需要上下文连贯性：如撰写论文，各部分内容需相互支撑
                在回答过程中，后续问题可以利用之前问题的回答作为背景信息

            Answer individually(独立回答)
                并行处理：每个子问题独立检索信息并回答，不依赖其他子问题的答案
                模块化整合：最后将所有子问题的答案综合成最终回答
                关键机制: 每个子问题单独检索文档, 生成的答案存储在某个地方中, 最终统一整合
                适用场景:
                    -- 信息聚合类问题：如 "人工智能有哪些应用领域？"，各领域可独立介绍
                    -- 多视角分析：如评估产品优缺点，每个维度可单独分析
                    -- 需要快速并行处理：提高效率，适合子问题间无依赖关系的场景

            注意: 针对复杂场景, 一般会考虑两种方式混合使用
                混合策略：根据问题复杂度组合两种方法。例如：
                    先用递归回答处理核心子问题链
                    再用独立回答补充外围信息
                评估指标：通过实验对比两种策略在特定任务上的准确率、连贯性和效率

        6. 最后，将所有子问题的回答综合起来，生成一个完整的、结构化的最终回答
"""

# 1. 读取文档并进行分割
# 1.1 读取web文档
web_doc_loader = WebBaseLoader(web_path="https://lilianweng.github.io/posts/2023-06-23-agent/")
web_doc = web_doc_loader.load()

# 1.2 文档分割
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=50
)

web_doc_chunk = text_splitter.split_documents(documents=web_doc)


# 2. 使用向量数据库存放分割文档, 并生成检索器
# 2.1 使用向量数据库存放分割文档chunk
vector_db = Chroma.from_documents(documents=web_doc_chunk, embedding=OpenAIEmbeddings())

# 2.2 使用向量数据库声明检索器
retriever = vector_db.as_retriever()


# 3. 使用llm对用户的查询生成多个子问题(子查询)
# 3.1 定义使用llm分解用户查询的提示词
# 提示词模板指导 AI 助手将输入问题分解为多个可独立回答的子问题
sub_queries_template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):
"""
sub_queries_prompt = ChatPromptTemplate.from_template(sub_queries_template)

# 3.2 定义用于分解用户查询的llm和chain
decompose_llm = ChatOpenAI(temperature=0)

decompose_chain = (
    sub_queries_prompt | 
    decompose_llm |
    StrOutputParser() |
    (lambda x: x.split("\n"))
)

# 定义用户的查询
question = "What are the main components of an LLM-powered autonomous agent system?"
# 执行用户查询的分解
sub_questions = decompose_chain.invoke({"question": question})
# print(sub_questions)

# 3.3 使用LLM为用户查询分解为多个子问题(子查询)-考虑包含原始问题、背景 Q&A 对和相关上下文信息
# 注意上面是演示如何使用LLM将用户的查询分解为多个子查询, 但是分解多个子查询后, 还没包含原始问题、背景 Q&A 对和相关上下文

# 模板包含原始问题、背景 Q&A 对和相关上下文, 指导模型利用这些信息回答问题
qa_context_template = """Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
"""

qa_context_prompt = ChatPromptTemplate.from_template(qa_context_template)

def format_qa_pair(question, answer):
    """
        定义一个QA回答格式化函数, 为接下来使用llm回答多个子查询问题做准备
    """
    formatted_string = ""
    formatted_string += f"Question: {question}\n Answer: {answer}\n]\n"
    return formatted_string.strip()

# 定义用于回答多个子查询问题的llm
qa_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# 定义保存 子查询-子查询答案 的变量
q_a_pairs = ""



# 5. 使用 RAG 技术为每个子问题生成回答
# 5.1 Answer recursively
# 使用训练对每个子查询调用llm进行回答, 并且没有的子问题的回答, 都会参考之前子问题的回答
# 然后将每个 子查询-答案 保存起来
for q in question:
    sub_qa_rag_chain = (
        {"context": itemgetter("question") | retriever,
         "question": itemgetter("question"),
         "q_a_pairs": itemgetter("q_a_pairs")} |
         qa_context_prompt |
         qa_llm |
         StrOutputParser()
    )
    
    # 调用子问题回答链对子问题进行回答
    answer = sub_qa_rag_chain.invoke({"question": q, "q_a_pairs":q_a_pairs})
    # 将子问题 - 答案 按照之前的格式化函数进行格式
    q_a_pair = format_qa_pair(q, answer)
    # 将所有 子问题 - 答案 拼接起来
    q_a_pairs = q_a_pairs + "\n --- \n" + q_a_pair
    
# 5.2 Answer individually
# 对每个子查询调用文档检索器和llm生成回答

# 先拉起下官方提供的模版
prompt_rag = hub.pull('rlm/rag-prompt')

# 定义逐个对子问题利用文档检索 + llm 回答的函数
def retrieve_and_rag(question, prompt_rag, sub_question_generator_chain):
    """
        对每个子问题进行rag操作
    """

    # 使用子问题生成器对用户查询分解为多个子查询(子问题)
    sub_questions = sub_question_generator_chain.invoke({"question": question})

    # 定义一个存放对每一个子问题的回答的变量
    rag_results = []

    # 对每个子问题进行检索, 然后在回答
    for sub_question in sub_questions:
        # 对每个子问题的进行检索
        retrieve_docs = retriever.get_relevant_documents(sub_question)
        # 对每个子问题利用检索到问题和子问题一起输入给llm, 让llm回答
        answer = (prompt_rag | qa_llm | StrOutputParser()).invoke({"context": retrieve_docs, "question": sub_question})
        # 存放每个子问题的回答
        rag_results.append(answer)
    
    # 返回对每个子问题的回答和对应的子问题
    return rag_results, sub_questions


# 执行对每个子问题的检索和回答
answers, questions = retrieve_and_rag(question=question, prompt_rag=prompt_rag, sub_question_generator_chain=decompose_chain)


# 6. 最后将所有子问题的回答综合起来，生成一个完整的、结构化的最终回答
# 6.1 定义用于将子问题和答案 组合成上下文的函数, 便于输入给llm
def format_qa_conpose(questions, answers):
    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\n Answer {i}: {answer}\n\n"
    return formatted_string.strip()

# 使用上面的函数获取context
context = format_qa_conpose(questions=questions, answers=answers)

# 用于最后融合所有子问题答案的提示词
final_prompts = """Here is a set of Q+A pairs:
{context}
Use these to synthesize an answer to the question: {question}
"""

# 6.2 定义最后用于检索、回答 所有子问题的进行融合的chain
final_rag_chain = (
    final_prompts |
    qa_llm |
    StrOutputParser()
)

final_response = final_rag_chain.invoke({"context": context, "question": question})
print(final_response)

