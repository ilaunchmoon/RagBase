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


"""