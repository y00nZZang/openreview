import re
import os

def clean_title(title):
    """
    제목에서 공백, 숫자, 특수문자를 제거하고 소문자로 변환.

    Args:
        title (str): 원본 제목.

    Returns:
        str: 정리된 제목.
    """
    # 공백, 숫자, 특수문자를 제거
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

def extract_sections_from_file(file_path, output_dir, delimiter="###"):
    """
    파일에서 특정 문단('introduction', 'conclusion')을 추출하여 저장.

    Args:
        file_path (str): 입력 파일 경로.
        output_dir (str): 출력 파일을 저장할 디렉토리.
        delimiter (str): 문단을 나눌 기준 문자열.

    Returns:
        None
    """
    # 파일 읽기
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    
    # 문단 나누기
    sections = content.split(delimiter)

    # 추출할 키워드
    target_titles = ["introduction", "conclusion"]

    # 결과 저장
    extracted_sections = {}
    for title in target_titles:
        result = find_section_with_cleaned_title(sections, title)
        if result:
            extracted_sections[title] = result.strip()

    # 저장 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 파일로 저장
    for section, content in extracted_sections.items():
        file_path = os.path.join(output_dir, f"{section}.txt")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
        print(f"Saved: {file_path}")

if __name__ == "__main__":
    # 입력 파일 경로
    input_file = "/Users/janghanyoon/Documents/openreview-rag/final/parse_text/test.md"

    # 출력 디렉토리
    output_directory = "extracted_sections"

    # 실행
    extract_sections_from_file(input_file, output_directory, delimiter="###")