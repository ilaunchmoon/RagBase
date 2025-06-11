"""
    RAPTOR(Recursive Abstractive Processing for Tree-Organized Retrieval)的核心在于用递归聚类 + 摘要生成
    构建一个多层级树状索引结构，用以捕捉长文档的多尺度语义信息
    传统RAG往往只索引原始文档的固定若干个短chunk, 这对多步骤或全局理解的任务效果有限, 而RAPTOR通过生成高层摘要, 增强了对大语境的覆盖


    步骤
        1. 文档加载与切分
            将文档按段落或句子切分成多个初始chunk, 通常大小可控。确保不破坏句子边界
        2. 文档嵌入
            使用语言模型或向量嵌入工具将每个chunk转成向量
        3. 聚类
            基于向量相似度, 采用如GaussianMixtureModel(GMM)等方法对chunk进行聚类, 并用BIC选择最优簇数
        4. 抽象生成摘要
            对每一簇chunk, 用LMM生成摘要, 形成该簇的父节点内容
        5. 递归处理
            将摘要作为新“chunk”, 再次嵌入、聚类、摘要，依此递归，直到摘要节点数或层级达到设定限制
        6. 构建索引树与VectorStore
            将所有叶子节点(原始chunks)和 中间／高层摘要节点统一存入向量数据库, 供查询时使用

    应用场景
        处理长文档：书籍、长报告、法律文本、技术白皮书
        复杂问答与多步推理: 如Quiz回答、书籍理解、论文问答等
        领域特定知识库构建: 如内部流程文档、医疗护理手册等, RAPTOR帮助快速整个内容搜索和摘要理解
        多粒度检索系统：对既需事实性，又需主题性回答的系统尤其适合

"""
import umap  # 导入UMAP用于降维和聚类
import numpy as np  # 导入numpy用于数值计算
import pandas as pd  # 导入pandas用于数据处理
import matplotlib.pyplot as plt  # 导入matplotlib用于数据可视化
import tiktoken  # 导入tiktoken用于计算文本的token数量
from bs4 import BeautifulSoup as Soup  # 导入BeautifulSoup用于HTML解析
from sklearn.mixture import GaussianMixture  # 导入高斯混合模型用于聚类
from langchain_openai import ChatOpenAI  # 注释掉的OpenAI聊天模型导入
from langchain import hub  # 导入LangChain hub
from langchain_core.runnables import RunnablePassthrough  # 导入RunnablePassthrough
from langchain_community.vectorstores import Chroma  # 导入Chroma向量数据库
from typing import Dict, List, Optional, Tuple  # 导入类型提示相关模块
from langchain_openai import OpenAIEmbeddings  # 导入OpenAI嵌入模型
from langchain.prompts import ChatPromptTemplate  # 导入聊天提示模板
from langchain_core.output_parsers import StrOutputParser  # 导入字符串输出解析器
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 导入递归字符文本分割器
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader  # 导入递归URL加载器用于爬取网页文档


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


# 统计文档中的token数
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)  # 获取指定的token编码方式
    num_tokens = len(encoding.encode(string))  # 对字符串进行编码并计算token数量
    return num_tokens


# 获取LangChain关于LCEL 的文档
url = "https://python.langchain.com/docs/expression_language/"  # 定义要爬取的LangChain Expression Language文档URL
loader = RecursiveUrlLoader(
    url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text  # 创建递归URL加载器，设置最大深度为20，并使用BeautifulSoup提取HTML文本内容
)
docs = loader.load()  # 加载文档内容

# 获取LangChain关于LCEL 的文档LCEL中关于
# LCEL 与 PydanticOutputParser 结合的文档, 递归深度为 1，仅爬取当前页面
url = "https://python.langchain.com/docs/modules/model_io/output_parsers/quick_start"  # 定义要爬取的PydanticOutputParser相关文档URL
loader = RecursiveUrlLoader(
    url=url, max_depth=1, extractor=lambda x: Soup(x, "html.parser").text  # 创建递归URL加载器，设置最大深度为1
)
docs_pydantic = loader.load()  # 加载PydanticOutputParser相关文档

# LCEL 与 Self Query 结合的文档：递归深度为 1，仅爬取当前页面
url = "https://python.langchain.com/docs/modules/data_connection/retrievers/self_query/"  # 定义要爬取的Self Query相关文档URL
loader = RecursiveUrlLoader(
    url=url, max_depth=1, extractor=lambda x: Soup(x, "html.parser").text  # 创建递归URL加载器，设置最大深度为1
)
docs_sq = loader.load()  # 加载Self Query相关文档

# 将上面爬取的3个文档合并起啦
docs.extend([*docs_pydantic, *docs_sq])  # 将所有文档合并到一个列表中
docs_texts = [d.page_content for d in docs]  # 提取所有文档的文本内容

# 计算每个文档的token数量
counts = [num_tokens_from_string(d, "cl100k_base") for d in docs_texts]  # 计算每个文档的token数量，使用cl100k_base编码（OpenAI的gpt-4等模型使用的编码）

# Plotting the histogram of token counts
plt.figure(figsize=(10, 6))  # 创建一个图形，设置大小为10x6英寸
plt.hist(counts, bins=30, color="blue", edgecolor="black", alpha=0.7)  # 绘制token数量的直方图，设置30个区间，蓝色柱状图，黑色边框，透明度0.7
plt.title("Histogram of Token Counts")  # 设置图表标题
plt.xlabel("Token Count")  # 设置x轴标签
plt.ylabel("Frequency")  # 设置y轴标签
plt.grid(axis="y", alpha=0.75)  # 添加y轴网格线，透明度0.75

# Display the histogram
# plt.show()  # 显示直方图


# 将合并的文档按照来源进行排序
d_sorted = sorted(docs, key=lambda x: x.metadata["source"])  # 按照文档的源URL对文档进行排序
d_reversed = list(reversed(d_sorted))  # 反转排序后的文档列表
concatenated_content = "\n\n\n --- \n\n\n".join(
    [doc.page_content for doc in d_reversed]  # 将所有文档内容用分隔符连接成一个字符串
)
print(
    "Num tokens in all context: %s"
    % num_tokens_from_string(concatenated_content, "cl100k_base")  # 打印所有文档内容的总token数量
)


# 进行文档风格
chunk_size_tok = 2000  # 设置文本块大小为2000个token
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=chunk_size_tok, chunk_overlap=0  # 创建基于tiktoken编码器的文本分割器，设置块大小和重叠部分
)
texts_split = text_splitter.split_text(concatenated_content)  # 对连接后的文档内容进行分割



embd = OpenAIEmbeddings()  # 初始化OpenAI嵌入模型实例
model = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")  # 注释掉的OpenAI聊天模型初始化


RANDOM_SEED = 224  # 设置随机种子，确保结果可复现

### --- Code from citations referenced above (added comments and docstrings) --- ###


def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Perform global dimensionality reduction on the embeddings using UMAP.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - dim: The target dimensionality for the reduced space.
    - n_neighbors: Optional; the number of neighbors to consider for each point.
                   If not provided, it defaults to the square root of the number of embeddings.
    - metric: The distance metric to use for UMAP.

    Returns:
    - A numpy array of the embeddings reduced to the specified dimensionality.

    使用UMAP对嵌入执行全局降维。

    参数:
    —embeddings: 以numpy数组的形式输入embeddings
    - dim: 约简空间的目标维数
    —n_neighbors: 可选; 每个点要考虑的邻居的数量
    如果未提供，则默认为嵌入数的平方根
    —metric: UMAP使用的距离度量值

    返回:
    -简化到指定维数的嵌入的numpy数组
    """
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)  # 如果未指定邻居数量，使用样本数的平方根作为默认值
    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric  # 使用UMAP算法进行全局降维
    ).fit_transform(embeddings)


def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    """
    Perform local dimensionality reduction on the embeddings using UMAP, typically after global clustering.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - dim: The target dimensionality for the reduced space.
    - num_neighbors: The number of neighbors to consider for each point.
    - metric: The distance metric to use for UMAP.

    Returns:
    - A numpy array of the embeddings reduced to the specified dimensionality.

    通常在全局聚类之后, 使用UMAP对嵌入执行局部降维。

    参数:
    —embeddings: 以numpy数组的形式输入embeddings
    - dim: 约简空间的目标维数
    —num_neighbors: 每个点要考虑的邻居数量
    —metric: UMAP使用的距离度量值

    返回:
    -简化到指定维数的嵌入的numpy数组
    """
    return umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric  # 使用UMAP算法进行局部降维
    ).fit_transform(embeddings)


def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    """
    Determine the optimal number of clusters using the Bayesian Information Criterion (BIC) with a Gaussian Mixture Model.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - max_clusters: The maximum number of clusters to consider.
    - random_state: Seed for reproducibility.

    Returns:
    - An integer representing the optimal number of clusters found.

    使用贝叶斯信息准则(BIC)和高斯混合模型确定最优簇数

    参数:
    —embeddings: 以numpy数组的形式输入embeddings
    —max_clusters: 需要考虑的最大集群数
    —random_state: 可复制的种子

    返回:
    —一个整数，表示找到的最优簇数
    """
    max_clusters = min(max_clusters, len(embeddings))  # 确保最大聚类数不超过样本数
    n_clusters = np.arange(1, max_clusters)  # 创建从1到max_clusters-1的数组
    bics = []  # 初始化BIC值列表
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)  # 创建高斯混合模型
        gm.fit(embeddings)  # 拟合模型
        bics.append(gm.bic(embeddings))  # 计算并存储BIC值
    return n_clusters[np.argmin(bics)]  # 返回使BIC值最小的聚类数


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    """
    Cluster embeddings using a Gaussian Mixture Model (GMM) based on a probability threshold.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - threshold: The probability threshold for assigning an embedding to a cluster.
    - random_state: Seed for reproducibility.

    Returns:
    - A tuple containing the cluster labels and the number of clusters determined.

    基于概率阈值的高斯混合模型(GMM)聚类嵌入

    参数:
    —embeddings: 以numpy数组的形式输入embeddings
    —threshold: 为集群分配嵌入的概率阈值
    —random_state: 可复制的种子

    返回:
    —包含集群标签和确定的集群数量的元组
    """
    n_clusters = get_optimal_clusters(embeddings)  # 获取最优聚类数
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)  # 创建高斯混合模型
    gm.fit(embeddings)  # 拟合模型
    probs = gm.predict_proba(embeddings)  # 计算每个样本属于各个聚类的概率
    labels = [np.where(prob > threshold)[0] for prob in probs]  # 根据阈值确定每个样本所属的聚类
    return labels, n_clusters  # 返回聚类标签和聚类数


def perform_clustering(
    embeddings: np.ndarray,
    dim: int,
    threshold: float,
) -> List[np.ndarray]:
    """
    Perform clustering on the embeddings by first reducing their dimensionality globally, then clustering
    using a Gaussian Mixture Model, and finally performing local clustering within each global cluster.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - dim: The target dimensionality for UMAP reduction.
    - threshold: The probability threshold for assigning an embedding to a cluster in GMM.

    Returns:
    - A list of numpy arrays, where each array contains the cluster IDs for each embedding.

    首先对嵌入进行全局降维，然后对其进行聚类
    使用高斯混合模型，最后在每个全局聚类中进行局部聚类

    参数:
    —embeddings: 以numpy数组的形式输入embeddings
    - dim: UMAP约简的目标维数
    —threshold: GMM中为集群分配嵌入的概率阈值
    """
    if len(embeddings) <= dim + 1:
        # Avoid clustering when there's insufficient data
        return [np.array([0]) for _ in range(len(embeddings))]  # 如果样本数不足，直接将所有样本归为一类

    # Global dimensionality reduction
    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)  # 对嵌入向量进行全局降维
    # Global clustering
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold  # 对全局降维后的向量进行聚类
    )

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]  # 初始化局部聚类结果列表
    total_clusters = 0  # 初始化总聚类数

    # Iterate through each global cluster to perform local clustering
    for i in range(n_global_clusters):
        # Extract embeddings belonging to the current global cluster
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])  # 提取属于当前全局聚类的嵌入向量
        ]

        if len(global_cluster_embeddings_) == 0:
            continue  # 如果当前全局聚类没有样本，跳过
        if len(global_cluster_embeddings_) <= dim + 1:
            # Handle small clusters with direct assignment
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]  # 对小聚类直接分配
            n_local_clusters = 1  # 设置局部聚类数为1
        else:
            # Local dimensionality reduction and clustering
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim  # 对当前全局聚类的嵌入向量进行局部降维
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold  # 对局部降维后的向量进行聚类
            )

        # Assign local cluster IDs, adjusting for total clusters already processed
        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])  # 提取属于当前局部聚类的嵌入向量
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)  # 找到这些嵌入向量在原始嵌入矩阵中的索引
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters  # 为这些样本分配局部聚类ID
                )

        total_clusters += n_local_clusters  # 更新总聚类数

    return all_local_clusters  # 返回所有样本的局部聚类结果


### --- Our code below --- ###


def embed(texts):
    """
    Generate embeddings for a list of text documents.

    This function assumes the existence of an `embd` object with a method `embed_documents`
    that takes a list of texts and returns their embeddings.

    Parameters:
    - texts: List[str], a list of text documents to be embedded.

    Returns:
    - numpy.ndarray: An array of embeddings for the given text documents.

    为文本文档列表生成嵌入

    这个函数假设存在一个带有方法‘ embed_documents ’的‘ embd ’对象
    它接受文本列表并返回它们的嵌入

    参数:
    - text: List[str]，要嵌入的文本文档列表

    返回:
    ——numpy。narray: 给定文本文档的嵌入数组
    """
    text_embeddings = embd.embed_documents(texts)  # 使用嵌入模型生成文本嵌入向量
    text_embeddings_np = np.array(text_embeddings)  # 将嵌入向量转换为numpy数组
    return text_embeddings_np


def embed_cluster_texts(texts):
    """
    Embeds a list of texts and clusters them, returning a DataFrame with texts, their embeddings, and cluster labels.

    This function combines embedding generation and clustering into a single step. It assumes the existence
    of a previously defined `perform_clustering` function that performs clustering on the embeddings.

    Parameters:
    - texts: List[str], a list of text documents to be processed.

    Returns:
    - pandas.DataFrame: A DataFrame containing the original texts, their embeddings, and the assigned cluster labels.

    嵌入文本列表并对其进行聚类, 返回一个包含文本、文本嵌入和聚类标签的DataFrame

    该函数将嵌入生成和聚类结合到一个步骤中。它假设存在
    之前定义的‘ perform_clustering ’函数，该函数对嵌入执行聚类

    参数:
    - texts: List[str]，要处理的文本文档列表

    返回:
    ——panda.DataFrame: 一个包含原始文本、它们的嵌入和分配的集群标签的DataFrame
    """
    text_embeddings_np = embed(texts)  # 生成文本嵌入向量
    cluster_labels = perform_clustering(
        text_embeddings_np, 10, 0.1  # 对嵌入向量进行聚类，使用10维降维，阈值为0.1
    )
    df = pd.DataFrame()  # 创建空DataFrame
    df["text"] = texts  # 添加文本列
    df["embd"] = list(text_embeddings_np)  # 添加嵌入向量列
    df["cluster"] = cluster_labels  # 添加聚类标签列
    return df


def fmt_txt(df: pd.DataFrame) -> str:
    """
    Formats the text documents in a DataFrame into a single string.

    Parameters:
    - df: DataFrame containing the 'text' column with text documents to format.

    Returns:
    - A single string where all text documents are joined by a specific delimiter.

    将DataFrame中的文本文档格式化为单个字符串

    参数:
    - df: 包含要格式化的文本文档的'text'列的DataFrame

    返回:
    -单个字符串，其中所有文本文档由特定分隔符连接
    """
    unique_txt = df["text"].tolist()  # 将文本列转换为列表
    return "--- --- \n --- --- ".join(unique_txt)  # 使用特定分隔符连接所有文本


def embed_cluster_summarize_texts(
    texts: List[str], level: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Embeds, clusters, and summarizes a list of texts. This function first generates embeddings for the texts,
    clusters them based on similarity, expands the cluster assignments for easier processing, and then summarizes
    the content within each cluster.

    Parameters:
    - texts: A list of text documents to be processed.
    - level: An integer parameter that could define the depth or detail of processing.

    Returns:
    - Tuple containing two DataFrames:
      1. The first DataFrame (`df_clusters`) includes the original texts, their embeddings, and cluster assignments.
      2. The second DataFrame (`df_summary`) contains summaries for each cluster, the specified level of detail,
         and the cluster identifiers.

    嵌入、聚类和总结文本列表。这个函数首先为文本生成嵌入，
    基于相似性对它们进行聚类，扩展聚类分配以便于处理，然后进行总结
    每个集群中的内容。

    参数:
    —texts: 要处理的文本文档列表
    - level: 一个整型参数, 可以定义处理的深度或细节

    返回:
        包含两个dataframe的元组:
        1. 第一个DataFrame(df_clusters)包括原始文本、它们的嵌入和集群分配
        2. 第二个DataFrame(df_summary)包含每个集群的摘要，指定的详细级别, 还有集群标识符
    """

    # Embed and cluster the texts, resulting in a DataFrame with 'text', 'embd', and 'cluster' columns
    df_clusters = embed_cluster_texts(texts)  # 嵌入并聚类文本

    # Prepare to expand the DataFrame for easier manipulation of clusters
    expanded_list = []  # 初始化扩展列表

    # Expand DataFrame entries to document-cluster pairings for straightforward processing
    for index, row in df_clusters.iterrows():
        for cluster in row["cluster"]:
            expanded_list.append(
                {"text": row["text"], "embd": row["embd"], "cluster": cluster}  # 为每个文本-聚类对创建一个条目
            )

    # Create a new DataFrame from the expanded list
    expanded_df = pd.DataFrame(expanded_list)  # 创建扩展后的DataFrame

    # Retrieve unique cluster identifiers for processing
    all_clusters = expanded_df["cluster"].unique()  # 获取所有唯一的聚类ID

    print(f"--Generated {len(all_clusters)} clusters--")  # 打印生成的聚类数量

    # Summarization
    template = """Here is a sub-set of LangChain Expression Language doc. 
    
    LangChain Expression Language provides a way to compose chain in LangChain.
    
    Give a detailed summary of the documentation provided.
    
    Documentation:
    {context}
    """  # 定义摘要生成的提示模板
    prompt = ChatPromptTemplate.from_template(template)  # 创建聊天提示模板
    chain = prompt | model | StrOutputParser()  # 创建摘要生成链：提示模板 -> 语言模型 -> 字符串输出解析器

    # Format text within each cluster for summarization
    summaries = []  # 初始化摘要列表
    for i in all_clusters:
        df_cluster = expanded_df[expanded_df["cluster"] == i]  # 获取当前聚类的所有文本
        formatted_txt = fmt_txt(df_cluster)  # 格式化文本
        summaries.append(chain.invoke({"context": formatted_txt}))  # 生成摘要并添加到列表

    # Create a DataFrame to store summaries with their corresponding cluster and level
    df_summary = pd.DataFrame(
        {
            "summaries": summaries,
            "level": [level] * len(summaries),  # 添加处理级别列
            "cluster": list(all_clusters),  # 添加聚类ID列
        }
    )

    return df_clusters, df_summary  # 返回聚类结果和摘要结果


def recursive_embed_cluster_summarize(
    texts: List[str], level: int = 1, n_levels: int = 3
) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Recursively embeds, clusters, and summarizes texts up to a specified level or until
    the number of unique clusters becomes 1, storing the results at each level.

    Parameters:
    - texts: List[str], texts to be processed.
    - level: int, current recursion level (starts at 1).
    - n_levels: int, maximum depth of recursion.

    Returns:
    - Dict[int, Tuple[pd.DataFrame, pd.DataFrame]], a dictionary where keys are the recursion
      levels and values are tuples containing the clusters DataFrame and summaries DataFrame at that level.

      递归地嵌入、聚集和总结文本, 直至指定级别或直到唯一集群的数量变为1, 将结果存储在每个级别

        参数:
            —texts: 列表[str]，表示要处理的文本
            - level: int, 当前递归级别(从1开始)
            - n_levels: int, 最大递归深度

        返回:
            - Dict[int, Tuple[pd. d]。DataFrame pd。DataFrame]]，其中键是递归的字典
            级别和值是包含该级别的集群DataFrame和汇总DataFrame的元组
    """
    results = {}  # 初始化结果字典

    # Perform embedding, clustering, and summarization for the current level
    df_clusters, df_summary = embed_cluster_summarize_texts(texts, level)  # 嵌入、聚类并摘要当前级别的文本

    # Store the results of the current level
    results[level] = (df_clusters, df_summary)  # 将当前级别的结果存入字典

    # Determine if further recursion is possible and meaningful
    unique_clusters = df_summary["cluster"].nunique()  # 获取当前级别的唯一聚类数
    if level < n_levels and unique_clusters > 1:  # 如果未达到最大级别且有多个聚类
        # Use summaries as the input texts for the next level of recursion
        new_texts = df_summary["summaries"].tolist()  # 将当前级别的摘要作为下一级的输入文本
        next_level_results = recursive_embed_cluster_summarize(
            new_texts, level + 1, n_levels  # 递归处理下一级
        )

        # Merge the results from the next level into the current results dictionary
        results.update(next_level_results)  # 将下一级的结果合并到结果字典

    return results  # 返回所有级别的结果


leaf_texts = docs_texts
results = recursive_embed_cluster_summarize(leaf_texts, level=1, n_levels=3)

# Initialize all_texts with leaf_texts
all_texts = leaf_texts.copy()  # 复制叶级文本到all_texts（注意：leaf_texts未在提供的代码中定义）

# Iterate through the results to extract summaries from each level and add them to all_texts
for level in sorted(results.keys()):  # 遍历所有级别
    # Extract summaries from the current level's DataFrame
    summaries = results[level][1]["summaries"].tolist()  # 获取当前级别的所有摘要
    # Extend all_texts with the summaries from the current level
    all_texts.extend(summaries)  # 将摘要添加到all_texts

# Now, use all_texts to build the vectorstore with Chroma
vectorstore = Chroma.from_texts(texts=all_texts, embedding=embd)  # 使用所有文本和嵌入模型创建Chroma向量存储
retriever = vectorstore.as_retriever()  # 创建检索器


# Prompt
prompt = hub.pull("rlm/rag-prompt")  # 从LangChain hub获取RAG提示模板

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)  # 格式化文档内容，用空行连接

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}  # 创建RAG链：检索相关文档并格式化，同时传递问题
    | prompt  # 应用提示模板
    | model  # 应用语言模型
    | StrOutputParser()  # 解析输出为字符串
)

# Question
response = rag_chain.invoke("How to define a RAG chain? Give me a specific code example.")  # 使用RAG链回答问题：如何定义RAG链并给出具体代码示例
print(response)

