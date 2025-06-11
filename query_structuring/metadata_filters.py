import datetime     #  # 导入datetime模块用于处理日期和时间
from typing import Literal, Optional, Tuple  # 导入类型注解，用于增强代码可读性和静态类型检
from langchain_openai import ChatOpenAI 
from pydantic import BaseModel, Field  # 导入LangChain和Pydantic的基础模型类和字段装饰器
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import YoutubeLoader

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

"""
    Query structuring之metadata_filters的元数据过滤
    先将自然语言查询转换为结构化的、可执行的查询语句(如 SQL、API 参数等), 便于数据库或系统精准检索
    再利用数据的元信息(如创建时间、类型、长度、标签等)作为过滤条件，缩小查询范围，提升结果相关性
    
    步骤:

        0. 加载文档

        1. 定义元数据过滤模型
            明确字段：定义系统支持的元数据过滤字段（如时间、数量、类别等）
            数据类型约束：为每个字段指定类型（如datetime、int、str）和验证规则
        
        2. 设计提示模板
            系统指令：告诉 LLM 将用户问题转换为结构化查询
            上下文约束：明确可用的元数据字段及其含义
        
        3. 调用LLM对用户查询进行自然语言解析, 一般都会结合LLM的function-calling
            LLM 调用：将用户问题和提示模板输入 LLM
            条件提取：LLM 从自然语言中识别元数据过滤条件（如时间、数量、比较关系）

        4. 结构化转换
            类型映射：将 LLM 输出的字符串转换为模型所需的数据类型（如字符串→日期、数值
            边界处理：处理隐含的边界条件（如 "2023 年" 自动转换为 "2023-01-01 至 2023-12-31"）
            验证与修正：检查转换后的字段是否符合模型定义（如日期格式是否合法）
        
        5.查询构建与执行
            组合条件：将结构化的过滤条件与内容检索条件（如关键词匹配）组合
            生成查询语句：转换为数据库可执行的查询（如 SQL、Elasticsearch DSL）

        6.结果过滤与返回
            应用元数据过滤：数据库根据结构化条件筛选结果
            结果排序：根据元数据字段（如发布时间、相关性）排序
            结果聚合：统计符合条件的记录数、平均值等（如 "符合条件的视频共 5 个"）


    适用场景
        多媒体内容检索: 根据发布时间、时长、播放量等筛选内容
        文档与知识库搜索: 按发表年份、引用次数、期刊级别等检索文献 或 按文档更新时间、作者、部门等筛选内部资料
        商品筛选：按价格区间、上架时间、销量、评分等过滤商品(如 "最近一周上架的、销量过千的电子产品")
        内容推荐：结合用户偏好和内容元数据(如 "推荐 2023 年发布的、播放量超过 10 万的机器学习教程")
"""


# 0. 加载文档
video_doc = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=pbAd8O1Lvm4", 
    add_video_info=True     # 声明要读取到视频的信息
)


# 1. 定义元数据过滤模型
# 这里定义了一个视频数据类型
# 包含如下两类搜索条件:
# 文本搜索：针对视频内容 (content_search) 和标题 (title_search)
# 过滤条件：视图数量、发布日期、视频长度等
# 过滤条件字段有: 
#   观看次数范围（min_view_count/max_view_count）
#   发布日期范围（earliest_publish_date/latest_publish_date）
#   视频长度范围（min_length_sec/max_length_sec）
class TutorialSearch(BaseModel):  # 定义TutorialSearch类，继承自Pydantic的BaseModel，用于结构化表示视频搜索查询
    """Search over a database of tutorial videos about a software library."""  # 类文档字符串，说明该类用于搜索软件库教程视频数据库

    content_search: str = Field(  # 定义content_search字段，类型为字符串
        ...,  # 表示该字段是必需的（必填项）
        description="Similarity search query applied to video transcripts.",  # 字段描述：应用于视频字幕的相似性搜索查询
    )
    title_search: str = Field(  # 定义title_search字段，类型为字符串
        ...,  # 表示该字段是必需的（必填项）
        description=(  # 字段描述：
            "Alternate version of the content search query to apply to video titles. "  # 应用于视频标题的内容搜索查询的替代版本
            "Should be succinct and only include key words that could be in a video "  # 应简洁，只包含可能出现在视频标题中的关键词
            "title."  # 标题中
        ),
    )
    min_view_count: Optional[int] = Field(  # 定义min_view_count字段，类型为可选整数
        None,  # 默认值为None
        description="Minimum view count filter, inclusive. Only use if explicitly specified.",  # 字段描述：最小观看次数过滤，包含边界值。仅在明确指定时使用
    )
    max_view_count: Optional[int] = Field(  # 定义max_view_count字段，类型为可选整数
        None,  # 默认值为None
        description="Maximum view count filter, exclusive. Only use if explicitly specified.",  # 字段描述：最大观看次数过滤，不包含边界值。仅在明确指定时使用
    )
    earliest_publish_date: Optional[datetime.date] = Field(  # 定义earliest_publish_date字段，类型为可选的datetime.date对象
        None,  # 默认值为None
        description="Earliest publish date filter, inclusive. Only use if explicitly specified.",  # 字段描述：最早发布日期过滤，包含边界值。仅在明确指定时使用
    )
    latest_publish_date: Optional[datetime.date] = Field(  # 定义latest_publish_date字段，类型为可选的datetime.date对象
        None,  # 默认值为None
        description="Latest publish date filter, exclusive. Only use if explicitly specified.",  # 字段描述：最新发布日期过滤，不包含边界值。仅在明确指定时使用
    )
    min_length_sec: Optional[int] = Field(  # 定义min_length_sec字段，类型为可选整数
        None,  # 默认值为None
        description="Minimum video length in seconds, inclusive. Only use if explicitly specified.",  # 字段描述：最小视频长度（秒），包含边界值。仅在明确指定时使用
    )
    max_length_sec: Optional[int] = Field(  # 定义max_length_sec字段，类型为可选整数
        None,  # 默认值为None
        description="Maximum video length in seconds, exclusive. Only use if explicitly specified.",  # 字段描述：最大视频长度（秒），不包含边界值。仅在明确指定时使用
    )

    def pretty_print(self) -> None:  # 定义pretty_print方法，无返回值
        for field in self.__fields__:  # 遍历模型的所有字段
            if getattr(self, field) is not None and getattr(self, field) != getattr(  # 检查字段值是否不为None且不等于默认值
                self.__fields__[field], "default", None  # 获取字段的默认值，如果不存在则返回None
            ):
                print(f"{field}: {getattr(self, field)}")  # 打印字段名和字段值

# 2. 设计提示模板
system = """You are an expert at converting user questions into database queries. \
You have access to a database of tutorial videos about a software library for building LLM-powered applications. \
Given a question, return a database query optimized to retrieve the most relevant results.

If there are acronyms or words you are not familiar with, do not try to rephrase them."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# 3. 调用LLM对用户查询进行自然语言解析
llm = ChatOpenAI(model="gpt-4", temperature=0)
# 3.1 利用LLM结合function-calling 进行进查询结构化解析
structured_llm = llm.with_structured_output(TutorialSearch)
# 调用解析链
query_analyzer = prompt | structured_llm

# 4. 调用示例
result = query_analyzer.invoke({"question": "rag from scratch"})
print("结构化查询结果:")
result.pretty_print()


