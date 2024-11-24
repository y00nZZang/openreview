from pymilvus import connections, Collection
import mysql.connector
from openai import OpenAI
from pydantic import BaseModel
import json
import tiktoken

# 데이터베이스 설정
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '122705',
    'database': 'openreview',
}

# OpenAI 설정
OPENAI_API_KEY = "OPENAI_API_KEY"
client = OpenAI(api_key=OPENAI_API_KEY)

# Milvus 컬렉션 이름 및 연결 설정
collection_name = "forum_embeddings"
connections.connect("default", host="localhost", port="19530")

class Score(BaseModel):
    score: int

def search_embeddings(query_embedding, field_name, top_k=10):
    """
    Milvus에서 유사한 벡터 검색
    """
    collection = Collection(name=collection_name)
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }
    results = collection.search(
        data=[query_embedding],
        anns_field=field_name,
        param=search_params,
        limit=top_k
    )
    return results[0]  # 첫 번째 검색 결과 반환

def retrieve_relevant_ids(intro_embedding, concl_embedding, weight_intro=0.6, weight_concl=0.4, top_k=3):
    """
    introduction과 conclusion 임베딩을 결합하여 최종 유사도를 계산
    """
    intro_results = search_embeddings(intro_embedding, field_name="embedding_intro", top_k=top_k*4)
    concl_results = search_embeddings(concl_embedding, field_name="embedding_concl", top_k=top_k*4)

    combined_scores = {}
    for result in intro_results:
        combined_scores[result.id] = combined_scores.get(result.id, 0) + weight_intro * result.distance
    for result in concl_results:
        combined_scores[result.id] = combined_scores.get(result.id, 0) + weight_concl * result.distance

    sorted_ids = sorted(combined_scores.items(), key=lambda x: x[1])
    return [item[0] for item in sorted_ids[:top_k]]

def fetch_forum_data(ids):
    """
    데이터베이스에서 주어진 ID에 대한 데이터를 가져오기
    """
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor(dictionary=True)
    format_strings = ','.join(['%s'] * len(ids))
    query = f"""
    SELECT 
        f.id AS forum_id, f.introduction, f.conclusion, f.text AS forum_text,
        r.summary_of_the_review, r.recommendation
    FROM 
        forum f
    LEFT JOIN 
        review r ON f.id = r.forum
    WHERE 
        f.id IN ({format_strings})
    """
    cursor.execute(query, ids)
    results = cursor.fetchall()
    cursor.close()
    connection.close()

    forum_data = {}
    for row in results:
        forum_id = row['forum_id']
        if forum_id not in forum_data:
            forum_data[forum_id] = {
                'id': forum_id,
                'introduction': row['introduction'],
                'conclusion': row['conclusion'],
                'text': row['forum_text'],
                'reviews': []
            }
        if row['summary_of_the_review']:
            forum_data[forum_id]['reviews'].append({
                'summary_of_the_review': row['summary_of_the_review'],
                'recommendation': row['recommendation']
            })
    return forum_data

def generate_prompt(forum_data, target_text, max_tokens=80000):
    """
    GPT를 위한 Few-shot Prompt 생성
    """
    prompt = "You are a helpful assistant evaluating research papers based on similar introductions and conclusions.\n"
    current_tokens = count_tokens(prompt)

    for forum_id, forum in forum_data.items():
        next_prompt = ""
        paper_text = f"Paper: {forum['text']}\n---------------------\n"
        next_prompt += paper_text
        current_tokens += count_tokens(paper_text)

        for review in forum['reviews']:
            if current_tokens >= max_tokens:
                break
            review_text = (
                f"Review Summary: {review['summary_of_the_review']}\n"
                f"Recommendation Score: {review['recommendation']}\n"
                "---------------------\n"
            )
            next_prompt += review_text
            if count_tokens(prompt + next_prompt) < max_tokens-1000:
                prompt += next_prompt
            else:
                break

    prompt += "\nNow evaluate the following paper:\n"
    prompt += f"Paper: {target_text}\n"
    prompt += "Provide your evaluation summary and recommendation score (0-10):"
    return prompt

def embed_text(text):
    """
    OpenAI를 사용해 텍스트를 임베딩
    """
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def basic_generate_score(target_text):
    """
    기본 GPT 모델을 사용하여 점수를 생성
    """
    prompt = f"""
    Paper: {target_text}
    Provide recommendation score (0-10):
    """
    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are a helpful assistant evaluating research papers."},
                {"role": "user", "content": prompt},
            ],
            response_format=Score,
        )
        response_content = response.choices[0].message.content
        response_json = json.loads(response_content)
        return response_json.get('score', None)
    except Exception as e:
        print(f"Error during basic GPT score generation: {e}")
        return None

def save_results_to_db(results):
    """
    생성된 점수를 데이터베이스에 저장
    """
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()

    create_table_query = """
    CREATE TABLE IF NOT EXISTS forum_score_results (
        id VARCHAR(255) PRIMARY KEY,
        avg_recommendation FLOAT,
        rag_score FLOAT,
        basic_score FLOAT   
    )
    """
    cursor.execute(create_table_query)

    insert_query = """
    INSERT INTO forum_score_results (id, avg_recommendation, rag_score, basic_score, difference)
    VALUES (%s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        avg_recommendation = VALUES(avg_recommendation),
        rag_score = VALUES(rag_score),
        basic_score = VALUES(basic_score)
    """
    cursor.executemany(insert_query, results)
    connection.commit()

    cursor.close()
    connection.close()

# 입력 토큰 수 계산
def count_tokens(text, model="gpt-4o-2024-08-06"):
    tokenizer = tiktoken.encoding_for_model(model)
    tokens = tokenizer.encode(text)
    return len(tokens)

def process_all_targets():
    """
    forum_data_split.is_vectordb = 0 인 모든 데이터를 처리
    """
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor(dictionary=True)

    query = """
    SELECT f.id, f.introduction, f.conclusion, f.text, f.average_recommendation
    FROM forum f
    JOIN forum_data_split fds ON f.id = fds.id
    WHERE fds.is_vectordb = 0
    """
    cursor.execute(query)
    all_targets = cursor.fetchall()

    results = []

    for target_data in all_targets:
        target_id = target_data['id']
        target_intro = target_data['introduction']
        target_concl = target_data['conclusion']
        target_text = target_data['text']
        target_avg_rec = target_data['average_recommendation']

        target_tokens = count_tokens(target_text)

        print(f"Processing ID: {target_id}")

        target_intro_embedding = embed_text(target_intro)
        target_concl_embedding = embed_text(target_concl)

        top_ids = retrieve_relevant_ids(target_intro_embedding, target_concl_embedding)
        forum_data = fetch_forum_data(top_ids)

        prompt = generate_prompt(forum_data, target_text, 128000 - target_tokens - 1000)

        try:
            response = client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant evaluating research papers."},
                    {"role": "user", "content": prompt},
                ],
                response_format=Score,
            )
            response_content = response.choices[0].message.content
            response_json = json.loads(response_content)
            rag_score = response_json.get('score', None)
        except Exception as e:
            print(f"Error during RAG GPT score generation for ID {target_id}: {e}")
            rag_score = None

        basic_score = basic_generate_score(target_text)

        print(f"Average Recommendation: {target_avg_rec}, RAG Score: {rag_score}, Basic Score: {basic_score}")

        # 결과를 즉시 데이터베이스에 삽입
        insert_query = """
        INSERT INTO forum_score_results (id, avg_recommendation, rag_score, basic_score)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            avg_recommendation = VALUES(avg_recommendation),
            rag_score = VALUES(rag_score),
            basic_score = VALUES(basic_score)
        """
        cursor.execute(insert_query, (target_id, target_avg_rec, rag_score, basic_score))

        connection.commit()

    cursor.close()
    connection.close()

if __name__ == "__main__":
    process_all_targets()