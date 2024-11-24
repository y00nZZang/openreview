import mysql.connector
from openai import OpenAI
from pydantic import BaseModel
import tiktoken  # OpenAI 토크나이저 라이브러리

class Score(BaseModel):
    score: int

# OpenAI API 키 설정
OPENAI_API_KEY = "OPENAI_API_KEY"
client = OpenAI(api_key=OPENAI_API_KEY)

# 데이터베이스 연결 설정
db_config = {
    "host": "localhost",        # 데이터베이스 호스트
    "user": "root",             # 데이터베이스 사용자 이름
    "password": "122705",       # 데이터베이스 비밀번호
    "database": "openreview"    # 데이터베이스 이름
}

# 데이터베이스에서 데이터 가져오기
def fetch_data(limit=5):
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor(dictionary=True)  # 결과를 딕셔너리로 반환
    cursor.execute("SELECT id, text, average_recommendation FROM forum LIMIT %s", (limit,))
    results = cursor.fetchall()
    cursor.close()
    connection.close()
    return results

# 입력 토큰 수 계산
def count_tokens(text, model="gpt-4"):
    tokenizer = tiktoken.encoding_for_model(model)
    tokens = tokenizer.encode(text)
    return len(tokens)

# GPT API 호출
def call_gpt_api(prompt):
    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are a helpful assistant to evaluate paper the score is between 0 and 10. The average of score is 5.4"},
                {"role": "user", "content": prompt}
            ],
            response_format=Score,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during GPT API call: {e}")
        return None

# 메인 실행 함수
def main():
    data = fetch_data(limit=5)  # 5개의 데이터 가져오기
    for record in data:
        record_id = record["id"]
        text_content = record["text"]
        real_score = record['average_recommendation']
        
        if text_content:  # text 필드가 비어있지 않은 경우에만 처리
            prompt = f"Please evaluate the following paper:\n\n{text_content}"
            token_count = count_tokens(prompt)
            print(f"Processing ID: {record_id}")
            print(f"Token count for ID {record_id}: {token_count}")
            
            # GPT API 호출
            score = call_gpt_api(prompt)
            if score:
                print(f"Score for ID {record_id}:\n{score}\n")
                print(f"Real score for ID {record_id}:\n{real_score}\n")
            else:
                print(f"Failed to generate score for ID {record_id}.\n")
        else:
            print(f"Text is empty for ID {record_id}.\n")

if __name__ == "__main__":
    main()