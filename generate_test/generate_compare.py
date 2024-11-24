import mysql.connector
from openai import OpenAI
from pydantic import BaseModel
import tiktoken  # OpenAI 토크나이저 라이브러리

# OpenAI API 키 설정
OPENAI_API_KEY = "OPENAI_API_KEY"
client = OpenAI(api_key=OPENAI_API_KEY)

# 데이터베이스 연결 설정
db_config = {
    "host": "localhost",        # 데이터베이스 호스트
    "user": "root",             # 데이터베이스 사용자 이름
    "password": "122705",  # 데이터베이스 비밀번호
    "database": "openreview"    # 데이터베이스 이름
}

class Score(BaseModel):
    review: str
    score: int

# 데이터베이스에서 데이터 가져오기
def fetch_data(limit=5):
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor(dictionary=True)
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

# GPT API 호출 - 방식 1
def call_gpt_api_v1(forum_data, target_paper):
    try:
        prompt = generate_few_shot_prompt(forum_data, target_paper)
        response = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for evaluating research papers."},
                {"role": "user", "content": prompt},
            ],
            response_format=Score,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during GPT API call v1: {e}")
        return None

# GPT API 호출 - 방식 2
def call_gpt_api_v2(text_content):
    try:
        prompt = f"Please evaluate the following paper:\n\n{text_content}"
        response = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are a helpful assistant to evaluate papers. The score is between 0 and 10. The average score is 5.4."},
                {"role": "user", "content": prompt}
            ],
            response_format=Score,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during GPT API call v2: {e}")
        return None

# Few-shot Prompt 생성
def generate_few_shot_prompt(forum_data, target_paper):
    prompt = "You are a helpful assistant evaluating research papers based on following papers and reviews.\n"
    for forum_id, data in forum_data.items():
        prompt += f"\nForum ID: {forum_id}\n"
        prompt += f"Forum Text: {data['forum_text']}\n"

        # 리뷰 섹션 추가
        prompt += "------ Reviews ------\n"
        for review in data['reviews']:
            prompt += f"Review Summary: {review['summary_of_the_review']}\n"
            prompt += f"Recommendation Score: {review['recommendation']}\n"
            prompt += "---------------------\n"

    prompt += "\nNow, evaluate the following paper:\n"
    prompt += f"Paper: {target_paper['text']}\n"
    prompt += "Provide your review summary and recommendation score (0-10):"
    return prompt

# 포럼과 연결된 리뷰 데이터 가져오기
def fetch_forum_with_reviews(limit=2):
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor(dictionary=True)
    query = """
    SELECT 
        f.id AS forum_id, f.text AS forum_text,
        r.summary_of_the_review, r.recommendation
    FROM 
        forum f
    LEFT JOIN 
        review r ON f.id = r.forum
    ORDER BY RAND()
    LIMIT %s
    """
    cursor.execute(query, (limit,))
    results = cursor.fetchall()
    cursor.close()
    connection.close()

    forum_data = {}
    for row in results:
        forum_id = row['forum_id']
        if forum_id not in forum_data:
            forum_data[forum_id] = {
                'forum_text': row['forum_text'],
                'reviews': []
            }
        if row['summary_of_the_review']:
            forum_data[forum_id]['reviews'].append({
                'summary_of_the_review': row['summary_of_the_review'],
                'recommendation': row['recommendation']
            })
    return forum_data

# 메인 실행 함수
def main():
    data = fetch_data(limit=5)
    forum_data = fetch_forum_with_reviews(limit=3)
    for record in data:
        record_id = record["id"]
        text_content = record["text"]
        real_score = record['average_recommendation']

        print(f"Processing ID: {record_id}")

        # 방식 1 점수 계산
        score_v1 = call_gpt_api_v1(forum_data, record)

        # 방식 2 점수 계산
        score_v2 = call_gpt_api_v2(text_content)

        # 결과 출력
        print(f"Real Score: {real_score}")
        print(f"Score V1: {score_v1}")
        print(f"Score V2: {score_v2}")
        #print(f"Difference (V1 - V2): {float(score_v1) - float(score_v2) if score_v1 and score_v2 else 'N/A'}\n")

if __name__ == "__main__":
    main()