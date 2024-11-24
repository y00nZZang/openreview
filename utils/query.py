from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType

# Milvus 연결 정보
COLLECTION_NAME = "forum_embeddings"

def create_index(collection):
    index_params_intro = {
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
        "metric_type": "L2"
    }
    index_params_concl = {
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
        "metric_type": "L2"
    }
    # embedding_intro 필드에 인덱스 생성
    collection.create_index(field_name="embedding_intro", index_params=index_params_intro)
    print(f"Index created for field 'embedding_intro' in collection '{COLLECTION_NAME}'.")

    # embedding_concl 필드에 인덱스 생성
    collection.create_index(field_name="embedding_concl", index_params=index_params_concl)
    print(f"Index created for field 'embedding_concl' in collection '{COLLECTION_NAME}'.")

def check_inserted_data():
    try:
        # Milvus 서버에 연결 설정
        connections.connect("default", host="localhost", port="19530")  # 호스트와 포트는 환경에 맞게 수정��세요.

        # 컬렉션 객체 생성
        collection = Collection(COLLECTION_NAME)

        # 인덱스 생성
        create_index(collection)

        # 데이터 개수 확인
        collection.flush()
        collection.load()  # 컬렉션 로드
        total_count = collection.num_entities
        print(f"Collection '{COLLECTION_NAME}' has {total_count} entities.")
        
        # 전체 데이터 개수 조회
        print(f"Total number of entities in the collection: {total_count}")
        
        # 데이터 확인 (샘플 데이터 가져오기)
        '''
        if total_count > 0:
            sample_query = collection.query(expr="", output_fields=["embedding_intro", "embedding_concl"], limit=5)
            print("Sample data from collection:")
            for idx, record in enumerate(sample_query, start=1):
                print(f"{idx}: {record}")

        else:
            print("No data found in the collection.")
        '''

    except Exception as e:
        print(f"Error while checking data in collection '{COLLECTION_NAME}': {str(e)}")

if __name__ == "__main__":
    check_inserted_data()