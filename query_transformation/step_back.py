from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate   # 用于支持少样本学习(few-shot learning)，通过提供示例来指导模型生成输出
from langchain_core.output_parsers import StrOutputParser  
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

"""
    演示使用Query-Transformations之Step Back将用户的查询后退一步的方式
    即通过抽象化约束条件、提升问题层级和保留核心意图，使检索更具包容性，最终提高 RAG 系统的回答质量

    Step Back操作步骤:
        1. 读取文档并进行分割
        2. 使用向量数据库存放分割文档, 并生成检索器
        3. 定义少样本示例提示词
        4. 使用llm对用户的查询进行step back改写
        5. 使用用户的原始查询和对应的step back查询, 借用检索器检索文档内容, 一起输入llm, 让llm生成用户查询的回答
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



# 3. 定义少样本示例提示词
examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel’s was born in what country?",
        "output": "what is Jan Sindel’s personal history?",
    },
]
# We now transform these to example messages
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# 4. 使用llm对用户的查询进行step back改写
# 4.1 定义使用llm进行用户查询step back的提示词
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
        ),
        # Few shot examples
        few_shot_prompt,
        # New question
        ("user", "{question}"),
    ]
)

#  4.2 使用llm对用户的查询进行step back 改写
generate_queries_step_back = prompt | ChatOpenAI(temperature=0) | StrOutputParser()
question = "What is task decomposition for LLM agents?"
generate_queries_step_back.invoke({"question": question})


#  5. 使用用户的原始查询和对应的step back查询, 借用检索器检索文档内容, 一起输入llm, 让llm生成用户查询的回答
# 5.1 定义用于最后生成答案的提示词模版
response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

# {normal_context}
# {step_back_context}

# Original Question: {question}
# Answer:"""
response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

# 5.1 定义链
chain = (
    {
        # Retrieve context using the normal question
        "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
        # Retrieve context using the step-back question
        "step_back_context": generate_queries_step_back | retriever,
        # Pass on the question
        "question": lambda x: x["question"],
    }
    | response_prompt
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
)

response = chain.invoke({"question": question})
print(response)

