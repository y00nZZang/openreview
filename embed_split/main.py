import mysql.connector
from openai import OpenAI
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '122705',
    'database': 'openreview',
}

# OpenAI 설정
OPENAI_API_KEY = "OPENAI_API_KEY"
client = OpenAI(api_key=OPENAI_API_KEY)

# Milvus 연결 설정
connections.connect("default", host="localhost", port="19530")
collection_name = "forum_embeddings"

def create_milvus_collection():
    """
    Milvus 컬렉션 생성
    """
    if not utility.has_collection(collection_name):
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=255, is_primary=True, auto_id=False),
            FieldSchema(name="embedding_intro", dtype=DataType.FLOAT_VECTOR, dim=1536),
            FieldSchema(name="embedding_concl", dtype=DataType.FLOAT_VECTOR, dim=1536)
        ]
        schema = CollectionSchema(fields, description="Forum embeddings with introduction and conclusion")
        collection = Collection(name=collection_name, schema=schema)
        print(f"Created collection: {collection_name}")
    else:
        collection = Collection(name=collection_name)
        print(f"Collection {collection_name} already exists.")
    return collection

def create_index(collection):
    """
    Milvus 컬렉션에 인덱스 생성
    """
    index_params = {
        "index_type": "IVF_FLAT",  # Index type
        "metric_type": "L2",      # Metric type
        "params": {"nlist": 128}  # Clustering parameters
    }
    collection.create_index(field_name="embedding_intro", index_params=index_params)
    collection.create_index(field_name="embedding_concl", index_params=index_params)
    print(f"Index created for collection: {collection.name}")
    collection.load()  # 검색 활성화
    print("Collection loaded for search.")

def embed_text(text):
    """
    OpenAI를 사용해 텍스트를 임베딩
    """
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def update_status(cursor, forum_id, status, error_message=None):
    """
    데이터베이스의 상태를 업데이트하거나 삽입
    """
    # 상태가 존재하는지 확인
    check_query = "SELECT COUNT(*) FROM forum_embedding_status WHERE id = %s"
    cursor.execute(check_query, (forum_id,))
    exists = cursor.fetchone()['COUNT(*)'] > 0

    if exists:
        # 존재하면 업데이트
        query = """
        UPDATE forum_embedding_status
        SET status = %s, error_message = %s
        WHERE id = %s
        """
        cursor.execute(query, (status, error_message, forum_id))
    else:
        # 존재하지 않으면 삽입
        query = """
        INSERT INTO forum_embedding_status (id, status, error_message)
        VALUES (%s, %s, %s)
        """
        cursor.execute(query, (forum_id, status, error_message))

def store_embeddings_in_milvus():
    """
    introduction과 conclusion 임베딩 생성 후 Milvus에 저장
    """
    collection = create_milvus_collection()
    collection.flush()
    create_index(collection)  # Index 생성 및 컬렉션 로드

    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor(dictionary=True)

    # VectorDB 대상 데이터 가져오기 (상태가 'pending'인 데이터만 처리)
    query = """
    SELECT f.id, f.introduction, f.conclusion
    FROM forum f
    JOIN forum_data_split s ON f.id = s.id
    LEFT JOIN forum_embedding_status es ON f.id = es.id
    WHERE s.is_vectordb = 1 AND (es.status IS NULL OR es.status != 'success')
    """
    cursor.execute(query)
    rows = cursor.fetchall()

    for row in rows:
        forum_id = row['id']
        try:
            # 임베딩 생성
            intro_embedding = embed_text(row['introduction'])
            concl_embedding = embed_text(row['conclusion'])

            # Milvus에 저장
            collection.insert([[forum_id], [intro_embedding], [concl_embedding]])

            # 상태 업데이트: 성공
            update_status(cursor, forum_id, 'success')
            connection.commit()  # 한 행 처리 후 바로 저장
            print(f"Processed ID: {forum_id}")

        except Exception as e:
            # 상태 업데이트: 실패
            error_message = str(e)
            update_status(cursor, forum_id, 'failed', error_message)
            connection.commit()  # 실패한 경우에도 상태 저장
            print(f"Error processing ID {forum_id}: {e}")

    cursor.close()
    connection.close()

store_embeddings_in_milvus()