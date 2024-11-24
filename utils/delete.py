from pymilvus import connections, Collection

# Milvus에 연결
connections.connect("default", host='localhost', port='19530')

# 삭제할 컬렉션 이름
collection_name = "forum_embeddings"

# 컬렉션 객체 생성
collection = Collection(name=collection_name)

# 컬렉션 삭제
collection.drop()

print(f"Collection '{collection_name}' has been deleted.")
