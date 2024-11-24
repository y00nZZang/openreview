import mysql.connector
from openai import OpenAI
import random
from pydantic import BaseModel
import tiktoken

# OpenAI API 키 설정
OPENAI_API_KEY = "OPENAI_API_KEY"
client = OpenAI(api_key=OPENAI_API_KEY)

# 데이터베이스 연결 설정
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "122705",
    "database": "openreview"
}


class Score(BaseModel):
    score: int

# 포럼과 연결된 모든 리뷰 가져오기
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

    # 포럼 별로 리뷰 그룹화
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

# 입력 토큰 수 계산
def count_tokens(text, model="gpt-4"):
    tokenizer = tiktoken.encoding_for_model(model)
    tokens = tokenizer.encode(text)
    return len(tokens)

# Few-shot Prompt 생성
def generate_few_shot_prompt(forum_data, target_paper):
    prompt = "You are a helpful assistant evaluating research papers based on prior reviews.\n"
    for forum_id, data in forum_data.items():
        prompt += f"\nForum ID: {forum_id}\n"
        prompt += f"Forum Text: {data['forum_text']}\n"  # 포럼 텍스트 일부 포함

        # 리뷰 섹션 추가
        prompt += "------ Reviews ------\n"
        for review in data['reviews']:
            prompt += f"Review Summary: {review['summary_of_the_review']}\n"
            prompt += f"Recommendation Score: {review['recommendation']}\n"
            prompt += "---------------------\n"

    # 타겟 논문 추가
    prompt += "\nNow, evaluate the following paper:\n"
    prompt += f"Text: {target_paper['text']}\n"  # 타겟 논문 본문 일부 포함
    prompt += "Provide your evaluation summary and recommendation score (0-10):"
    return prompt

# 랜덤 타겟 논문 가져오기
def fetch_random_target_paper():
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor(dictionary=True)
    query = "SELECT id, text, average_recommendation FROM forum ORDER BY RAND() LIMIT 1"
    cursor.execute(query)
    result = cursor.fetchone()
    cursor.close()
    connection.close()
    return result

# 메인 실행 함수
def main():
    # 포럼과 리뷰 데이터 가져오기
    forum_data = fetch_forum_with_reviews(limit=3)
    
    # 랜덤 타겟 논문 가져오기
    target_paper = fetch_random_target_paper()

    # Few-shot Prompt 생성
    few_shot_prompt = generate_few_shot_prompt(forum_data, target_paper)

    # 토큰 수 계산
    token_count = count_tokens(few_shot_prompt)
    print(f"Generated Few-shot Prompt:\n{few_shot_prompt}")
    print(f"Token Count: {token_count}")

    # GPT API 호출
    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for evaluating research papers."},
                {"role": "user", "content": few_shot_prompt},
            ],
            response_format=Score,
        )
        print(f"GPT Response:\n{response.choices[0].message.content}")
        print(f"Real score: {target_paper['average_recommendation']}")
    except Exception as e:
        print(f"Error during GPT API call: {e}")

if __name__ == "__main__":
    main()