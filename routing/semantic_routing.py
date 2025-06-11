from langchain.utils.math import cosine_similarity                                  # 用于计算向量间的余弦相似度，判断文本语义的相似程度
from langchain_core.runnables import RunnableLambda, RunnablePassthrough            # 允许将自定义函数转换为 LangChain 可运行组件; 直接传递输入数据，不做修改
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


"""
    Routing之semantic_routing(语义路由)
    指通过分析用户查询的语义特征，将其定向到最相关的数据源或处理路径的过程
    实现了从 "关键词匹配" 到 "语义理解" 的跨越，显著提升了复杂查询场景下的响应质量

    关键步骤:
        1. 查询语义编码
        2. 路由空间构建
        3. 语义相似度计算
        4. 路由决策生成
        5. 路由执行与反馈

    解决的问题:
        语义歧义问题	    同一关键词对应不同领域（如 “苹果” 指代水果或公司）	 通过语义向量空间的相似度计算消除歧义
        多源数据管理混乱	跨领域知识混合导致检索结果混杂	                    建立领域标签与语义空间的映射，实现精准分流
        计算资源浪费	    对所有查询使用统一处理流程，效率低下	            根据语义特征动态调用轻量级 / 重量级处理模块

        
    适用场景
        企业多部门知识管理：将销售查询路由至产品知识库，技术问题路由至研发文档
        学术研究助手：根据论文主题自动路由至不同学科数据库（物理、化学、生物）
        医疗问诊系统：将症状查询按科室（心内科、神经科）进行语义分流
        多语言客服系统：结合语义与语言特征，将查询路由至对应语言的客服知识库

    示例说明
        将用户问题转换为向量表示
        计算问题与不同领域提示模板的语义相似度
        自动选择最匹配的提示模板
        使用选定的模板调用 AI 模型生成回答
"""


# 定义了两个不同领域的提示模板，分别针对物理和数学
# 每个模板设定了 AI 助手的角色和回答风格
# {query}是一个占位符，将在运行时被用户问题替换
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{query}"""

math_template = """You are a very good mathematician. You are great at answering math questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.


Here is a question:
{query}"""

# 创建 OpenAI 嵌入模型实例
# 将两个提示模板放入列表
# 使用嵌入模型将提示模板转换为向量表示，便于后续计算相似度
embeddings = OpenAIEmbeddings()
prompt_templates = [physics_template, math_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)

# 定义使用语义相似度路由函数
def prompt_router(input):
    # 将用户问题转换为向量嵌入
    query_embedding = embeddings.embed_query(input["query"])
    # 计算问题与两个提示模板的余弦相似度
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    # 获取与用户查询相似度最高的提示词模板
    most_similar = prompt_templates[similarity.argmax()]
    # 选择相似度最高的提示模板 
    print("Using MATH" if most_similar == math_template else "Using PHYSICS")
    # 返回选中的提示模板对象
    return PromptTemplate.from_template(most_similar)


# 使用 LangChain 的管道操作符|构建处理流程
chain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(prompt_router)     #  应用路由函数，选择合适的提示模板
    | ChatOpenAI()  
    | StrOutputParser()
)

print(chain.invoke("What's a black hole"))