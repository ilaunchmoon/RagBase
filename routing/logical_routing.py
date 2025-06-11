from typing import Literal                                      # 用于定义固定值的类型注解
from langchain_core.prompts import ChatPromptTemplate  
from langchain_core.runnables import RunnableLambda         
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.pydantic_v1 import BaseModel, Field         # BaseModel 和 Field 用于创建数据模型

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


"""
    Routing之logical_routing(逻辑路由)
        基于预定义规则和结构进行检索的方法，它依赖于明确的元数据和组织结构
  

        1. 查询分析：解析用户查询，提取关键词、实体和逻辑关系
        2. 规则匹配：根据预定义的规则映射查询到特定的知识库分区
        3. 结构化检索：在选定的分区内执行结构化查询
        4. 结果合并：整合多个分区的结果，按相关性排序
        5. 反馈优化：根据用户反馈调整规则和权重


        适用场景
            结构化知识库

    演示示例如下:
        传入用户问题
        系统会自动：
            使用提示模板格式化问题
            调用 LLM 生成路由决策
            根据决策选择相应的处理链
            返回最终结果
"""
# 数据模型 RouteQuery，用于结构化输出
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    # datasource 字段使用 Literal 类型，限制只能是三种预定义的数据源之一
    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        ...,
        # Field 的描述文本指导 LLM 如何选择合适的数据源
        description="Given a user question choose which datasource would be most relevant for answering their question",
    )

# 利用LLM的FunctionCalling技术, 实现结构化输出的功能with_structured_ouput()这个就是FunctionCalling调用的函数
# with_structured_output 方法启用了结构化输出功能，让 LLM 按照 RouteQuery 模型的格式返回结果
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm = llm.with_structured_output(RouteQuery)

# 定义提示词
# 系统消息指导 LLM 作为路由专家工作
# 人类消息部分包含一个占位符 {question}，用于插入用户问题
system = """You are an expert at routing a user question to the appropriate data source.
Based on the programming language the question is referring to, route it to the relevant data source.
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# 定义可以生成结构化路由的决策chain
router = prompt | structured_llm

# 定义路由提示词模板: 提供了一个示例问题，包含 Python 代码
question = """Why doesn't the following code work:

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
prompt.invoke("french")
"""

# 调用路由链, 此时会按照提示词输出一个 RouteQuery 实例，包含选择的数据源
result = router.invoke({"question": question})

# 定义路由后逻辑
def choose_route(result):
    if "python_docs" in result.datasource.lower():
        ### Logic here 
        return "chain for python_docs"
    elif "js_docs" in result.datasource.lower():
        ### Logic here 
        return "chain for js_docs"
    else:
        ### Logic here 
        return "golang_docs"

# 定义输出后的链
final_chain = router | RunnableLambda(choose_route)
final_respose = final_chain.invoke({"question": question})
print(final_respose.datasource) # 预计输出python_docs
