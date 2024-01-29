import json
import os

import markdown
import psycopg2
from PyPDF2 import PdfFileReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import BartTokenizer

tok = BartTokenizer.from_pretrained("facebook/bart-large")


def read_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension == '.txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    elif file_extension == '.pdf':
        with open(file_path, 'rb') as file:
            pdf = PdfFileReader(file)
            return ' '.join(page.extractText() for page in pdf.pages)
    elif file_extension == '.md':
        with open(file_path, 'r') as file:
            md = file.read()
            return markdown.markdown(md)
    else:
        return None


def split_text_into_chunks(text, max_length=1000):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,
        chunk_overlap=50,
        length_function=lambda x: len(tok(x)["input_ids"]),
        is_separator_regex=False,
    )
    chunks = text_splitter.create_documents([text])

    enriched_chunks = []
    for i, chunk in enumerate(chunks):
        enriched_chunk = {
            'metadata': {
                'chunk_index': i,
                'start_position': i * max_length,
                'end_position': (i + 1) * max_length if (i + 1) * max_length < len(text) else len(text),
            },
            'content': chunk
        }
        enriched_chunks.append(enriched_chunk)

    return enriched_chunks


def read_files_in_folder(folder_path):
    file_names = os.listdir(folder_path)
    files = []
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        content = read_file(file_path)
        if content is not None:
            chunks = split_text_into_chunks(content)
            files.append({'file_name': file_name, 'content': chunks})
    return files


def indexar(content):
    # Conexão com o banco de dados PostgreSQL
    conn = psycopg2.connect(dbname='vectordb', user='testuser', password='testpwd', host='localhost', port=5432)
    cursor = conn.cursor()

    # Carregue o modelo pré-treinado
    model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

    # Inserção de registros e cálculo dos embeddings
    for content_details in content['content']:
        text = content_details['content'].page_content
        metadata = content_details['metadata']

        # Calcular o embedding da content_details
        texto_embedding = model.encode(text)

        # Convert the numpy array to a list
        texto_embedding_list = texto_embedding.tolist()

        # Converta o dicionário metadata para uma string JSON
        metadata_json = json.dumps(metadata)

        # Inserir o registro na tabela
        cursor.execute("INSERT INTO embeddings (text, source, metadata, embedding) VALUES (%s, %s, %s, %s);",
                       (text, content['file_name'], metadata_json, texto_embedding_list))

    # Commit das alterações e fechamento da conexão
    conn.commit()
    cursor.close()
    conn.close()


def buscar(query):
    # Conexão com o banco de dados PostgreSQL
    conn = psycopg2.connect(dbname='vectordb', user='testuser', password='testpwd', host='localhost', port=5432)
    cursor = conn.cursor()

    # Carregue o modelo pré-treinado
    model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

    # Obtenha o embedding para a consulta
    query_embedding = model.encode(query)

    # Convert the numpy array to a list
    query_embedding_list = query_embedding.tolist()

    # Convert the list to a string
    query_embedding_str = ', '.join(map(str, query_embedding_list))

    # Execute a consulta no banco de dados usando a função de similaridade do pgvector
    cursor.execute(
        f"SELECT text, embedding, (embedding <-> array[{query_embedding_str}]::vector) as distance FROM embeddings ORDER BY embedding <-> array[{query_embedding_str}]::vector LIMIT 3;")

    results = cursor.fetchall()

    # Imprima os resultados
    print(f'Consulta: "{query}"\n')
    for result in results:
        print(f'Text: "{result[0]}"')
        print(f'Distance: {result[2]}')
        print()

    # Feche a conexão com o banco de dados
    cursor.close()
    conn.close()


def run_indexar():
    contents = read_files_in_folder("docs")
    print(contents)
    for content in contents:
        indexar(content)


if __name__ == '__main__':
    run_indexar()
    buscar('O que é laranja?')
