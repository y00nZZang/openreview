from pymilvus import connections, Collection

# Milvus에 연결
connections.connect("default", host='localhost', port='19530')

# 컬렉션 이름 설정
collection_name = "forum_embeddings"

# 컬렉션 객체 생성
collection = Collection(collection_name)

# 컬렉션의 데이터 개수 확인
num_entities = collection.num_entities
print(f"Number of entities in the collection: {num_entities}")

# 데이터가 추가되었는지 확인
if num_entities > 0:
    print("데이터가 성공적으로 추가되었습니다.")
else:
    print("데이터가 추가되지 않았습니다.")