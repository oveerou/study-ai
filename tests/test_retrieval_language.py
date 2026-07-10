from ragbase.config import Config


def test_retrieval_models_support_chinese_course_materials():
    assert Config.Model.EMBEDDINGS == "BAAI/bge-small-zh-v1.5"
    assert Config.Model.RERANKER == "ms-marco-MultiBERT-L-12"
