from pymilvus import connections, Collection
import mysql.connector
from openai import OpenAI
from pydantic import BaseModel
import json

db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '122705',
    'database': 'openreview',
}

# OpenAI 설정
OPENAI_API_KEY = "OPENAI_API_KEY"
client = OpenAI(api_key=OPENAI_API_KEY)

# Milvus 컬렉션 이름
collection_name = "forum_embeddings"

# Milvus 연결 설정
connections.connect("default", host="localhost", port="19530")
collection_name = "forum_embeddings"


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
    # introduction 검색
    intro_results = search_embeddings(intro_embedding, field_name="embedding_intro", top_k=top_k*4)

    # conclusion 검색
    concl_results = search_embeddings(concl_embedding, field_name="embedding_concl", top_k=top_k*4)

    # 결과 결합
    combined_scores = {}
    for result in intro_results:
        combined_scores[result.id] = combined_scores.get(result.id, 0) + weight_intro * result.distance
    for result in concl_results:
        combined_scores[result.id] = combined_scores.get(result.id, 0) + weight_concl * result.distance

    # 유사도를 기준으로 정렬하여 상위 ID 반환
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
                'id': row['forum_id'],
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

def generate_prompt(forum_data, target_text):
    """
    GPT를 위한 Few-shot Prompt 생성
    """
    prompt = "You are a helpful assistant evaluating research papers based on similar introductions and conclusions.\n"
    for forum_id, forum in forum_data.items():  # 딕셔너리의 키와 값을 사용
        prompt += f"Paper: {forum['text']}\n"
        prompt += "---------------------\n"
        for review in forum['reviews']:
            prompt += f"Review Summary: {review['summary_of_the_review']}\n"
            prompt += f"Recommendation Score: {review['recommendation']}\n"
            prompt += "---------------------\n"

    # 타겟 데이터 추가
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

def rag_generate_score(target_text):
    """
    RAG 방식으로 점수를 생성
    """
    pass

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
                {"role": "system", "content": "You are a helpful assistant evaluating research papers. Now evaluate the following paper:"},
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
    
def compare_scores(target_text, target_avg_rec, rag_score, basic_score):
    """
    두 방식의 점수를 비교하고 출력
    """
    print("\n### Score Comparison ###")
    print(f"Target Average Recommendation: {target_avg_rec}")
    print(f"RAG Generated Score: {rag_score}")
    print(f"Basic GPT Generated Score: {basic_score}")
    if rag_score and basic_score:
        print(f"Difference (RAG - Basic): {rag_score - basic_score}")
        
def main():
    # 데이터베이스 연결
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor(dictionary=True)

    # 벡터 DB에 추가되지 않은 데이터를 쿼리
    query = """
    SELECT f.id, f.introduction, f.conclusion, f.text, f.average_recommendation
    FROM forum f 
    JOIN forum_data_split fds ON f.id = fds.id 
    WHERE fds.is_vectordb = 0 
    LIMIT 1
    """
    cursor.execute(query)
    target_data = cursor.fetchone()

    if not target_data:
        print("No target data available for processing.")
        cursor.close()
        connection.close()
        return

    # 타겟 데이터 설정
    target_id = target_data['id']
    target_intro = target_data['introduction']
    target_concl = target_data['conclusion']
    target_text = target_data['text']
    target_avg_rec = target_data['average_recommendation']
    # 타겟 텍스트 임베딩 생성
    target_intro_embedding = embed_text(target_intro)
    target_concl_embedding = embed_text(target_concl)

    # 유사한 ID 검색
    top_ids = retrieve_relevant_ids(target_intro_embedding, target_concl_embedding)

    # ID에 해당하는 데이터 가져오기
    forum_data = fetch_forum_data(top_ids)

    # 프롬프트 생성
    prompt = generate_prompt(forum_data, target_text)

    print("Generated Prompt:")
    print(prompt)

    print(f"Target id: {target_id}")
    print(f"Reference ids: {top_ids}")
    print(f"Target Average Recommendation: {target_avg_rec}")

    # OpenAI GPT 호출
    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for evaluating research papers."},
                {"role": "user", "content": prompt},
            ],
            response_format=Score,
        )

        try:
            response_content = response.choices[0].message.content
            response_json = json.loads(response_content)
            score = response_json.get('score', None)
            if score is not None:
                rag_score = score
                print(f"Generated Score: {score}")
            else:
                print("Score not found in the response.")
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")

    except Exception as e:
        print(f"Error during GPT API call: {e}")
    
    basic_score = basic_generate_score(target_text)
    print(f"Basic GPT Generated Score: {basic_score}")
    compare_scores(target_text, target_avg_rec, rag_score, basic_score)

    cursor.close()
    connection.close()

if __name__ == "__main__":
    main()