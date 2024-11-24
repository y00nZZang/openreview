import mysql.connector
import random

# MySQL 데이터베이스 연결 설정
db_config = {
    'host': 'localhost',  # 데이터베이스 호스트
    'user': 'root',  # 데이터베이스 사용자 이름
    'password': '122705',  # 데이터베이스 비밀번호
    'database': 'openreview',  # 데이터베이스 이름
}

def split_data():
    """
    데이터베이스에서 introduction과 conclusion이 비어있지 않은 행을 랜덤으로 95%와 5%로 분리.
    """
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor(dictionary=True)

    # introduction, conclusion이 비어있지 않은 행 조회
    query = "SELECT id FROM forum WHERE introduction IS NOT NULL AND conclusion IS NOT NULL"
    cursor.execute(query)
    rows = cursor.fetchall()

    # 데이터 랜덤 분리
    data = [row['id'] for row in rows]
    random.shuffle(data)
    split_index = int(len(data) * 0.95)
    vectordb_data = data[:split_index]
    test_data = data[split_index:]

    # 결과를 forum_data_split 테이블에 저장
    for forum_id in vectordb_data:
        cursor.execute("INSERT INTO forum_data_split (id, is_vectordb) VALUES (%s, TRUE)", (forum_id,))
    for forum_id in test_data:
        cursor.execute("INSERT INTO forum_data_split (id, is_vectordb) VALUES (%s, FALSE)", (forum_id,))

    connection.commit()
    cursor.close()
    connection.close()
    print(f"Data split complete: {len(vectordb_data)} for VectorDB, {len(test_data)} for testing.")

split_data()