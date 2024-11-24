import mysql.connector
import re

def clean_title(title):
    """
    제목에서 공백, 숫자, 특수문자를 제거하고 소문자로 변환.

    Args:
        title (str): 원본 제목.

    Returns:
        str: 정리된 제목.
    """
    cleaned_title = re.sub(r"[^\w]", "", title)  # 알파벳과 숫자만 남기기
    return cleaned_title.lower()

def find_section_with_cleaned_title(sections, title_keyword):
    """
    특정 제목 키워드를 포함하는 문단을 찾는 함수.

    Args:
        sections (list of str): 분리된 문단 목록.
        title_keyword (str): 찾고자 하는 문단 제목 키워드(예: 'introduction').

    Returns:
        str or None: 해당 키워드를 포함한 문단, 없으면 None.
    """
    cleaned_keyword = clean_title(title_keyword)  # 입력 키워드 정리

    for section in sections:
        # 문단의 첫 번째 줄 정리
        first_line = section.strip().splitlines()[0]
        cleaned_first_line = clean_title(first_line)
        if cleaned_keyword in cleaned_first_line:
            return section
    return None

def extract_sections_from_text(text, delimiter="###"):
    """
    텍스트에서 특정 섹션('introduction', 'conclusion')을 추출.

    Args:
        text (str): 원본 텍스트.
        delimiter (str): 문단을 나눌 기준 문자열.

    Returns:
        dict: 추출된 섹션('introduction', 'conclusion') 내용.
    """
    sections = text.split(delimiter)  # 문단 나누기

    # 추출할 키워드
    target_titles = ["introduction", "conclusion"]

    extracted_sections = {}
    for title in target_titles:
        result = find_section_with_cleaned_title(sections, title)
        if result:
            extracted_sections[title] = result.strip()
    return extracted_sections

def process_database():
    """
    데이터베이스의 모든 레코드에 대해 'introduction'과 'conclusion' 섹션을 추출하고 업데이트.
    """
    try:
        # 데이터베이스 연결
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='122705',
            database='openreview'
        )

        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)

            # 'text'가 NULL이 아닌 모든 레코드 선택
            cursor.execute("SELECT id, text FROM forum WHERE text IS NOT NULL")
            records = cursor.fetchall()

            for record in records:
                print(f"Processing record ID {record['id']}...")

                try:
                    text = record['text']
                    extracted_sections = extract_sections_from_text(text, delimiter="###")

                    # 추출된 'introduction'과 'conclusion' 섹션
                    introduction = extracted_sections.get('introduction', None)
                    conclusion = extracted_sections.get('conclusion', None)

                    # 데이터베이스 업데이트
                    update_query = """
                        UPDATE forum
                        SET introduction = %s, conclusion = %s
                        WHERE id = %s
                    """
                    cursor.execute(update_query, (introduction, conclusion, record['id']))
                except Exception as e:
                    print(f"Error processing record ID {record['id']}: {e}")
                    continue

            # 변경사항 커밋
            connection.commit()

    except mysql.connector.Error as e:
        print(f"Database error: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

if __name__ == "__main__":
    process_database()