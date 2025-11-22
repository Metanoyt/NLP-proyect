from datasets import load_dataset,Dataset
import faiss
from sentence_transformers import SentenceTransformer
try:
    from nltk.tokenize.toktok import ToktokTokenizer
except Exception:
    try:
        from nltk.tokenize import ToktokTokenizer
    except Exception:
        # Fallback simple tokenizer using regex split on non-word characters
        class ToktokTokenizer:
            def tokenize(self, text):
                return [t for t in re.split(r'\W+', text) if t]
import os
import re

wordTokenizer = ToktokTokenizer()


dataset_hf_id = ""
repo_base_path = "./sample_projects"
processing_base_path = "./pyfiles/"
original_files_path = "./original_files/"
embedding_model_name = "sentence-transformers/(something)"



def splitFileByBlockOfCode(file_txt: str) -> list:
    """
    Funcion para dividir un archivo python y recolectar los bloques de texto que lo componen, ignorando los imports.
    Entradas: El texto completo del archivo
    Salida: Una lista de bloques de codigo (funciones, clases, etc)
    """
    blocks = []
    current_block = []
    lines = file_txt.split('\n')
    inside_function_or_class = False

    for line in lines:
        stripped_line = line.strip()

        if stripped_line.startswith('import'):
            continue
        if stripped_line.startswith('from'):
            # Handle multi-line imports
            while not stripped_line.endswith('\\') and not stripped_line.endswith('import'):
                stripped_line += ' ' + next(lines).strip()
            continue

        if stripped_line.startswith('def ') or stripped_line.startswith('class '):
            if current_block:
                blocks.append('\n'.join(current_block))
                current_block = []
            inside_function_or_class = True

        if inside_function_or_class:
            current_block.append(line)

            if stripped_line == '' and current_block:
                blocks.append('\n'.join(current_block))
                current_block = []
                inside_function_or_class = False

    if current_block:
        blocks.append('\n'.join(current_block))

    return blocks


def cleanTextChunk(file_txt:str, tokenizer=wordTokenizer):
    """
    Funcion para limpiar un bloque de codigo de texto irrelevante

    Entradas: El texto completo del bloque de codigo
    Salida: El texto limpio del bloque de codigo
    """
    
    txt_no_special = re.sub(r"""[ 0-9 !?',".-<>(){}!\]:@%;&*/[/]""", " ", file_txt)
    tokens = tokenizer.tokenize(txt_no_special)  # Potentially can skip tokenization here

    no_useless_terms = removeUnwantedTerms(tokens) 
    preprocessed_text = ' '.join(no_useless_terms)

    return preprocessed_text


def removeUnwantedTerms(tokens):
        """
        Función para remover las stop words de un set de tokens
        Entradas: Un texto o cadena de carácteres
        Salida: Una lista de tokens
        """
        useless_terms = ['def', '#', 'return', 'if', 'for', 'in', 'and', 'or', 'not', 'is', 'to', 'the', 'a', 'an', 'of', 'on', 'with', 'as', 'by', 'this', 'that', 'it', 'from', 'at', 'be', 'are', 'was', 'were', 'but', 'else', 'elif', 'import', 'class', 'self']
        
        filtered_tokens = [token for token in tokens if token not in useless_terms]
        return filtered_tokens




def loadLocalDataset(path_folder) -> Dataset:
    data = {'file_name': [], 'text': [], 'important_terms': []}
    for root, dirs, files in os.walk(path_folder):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_text = f.read()
                    blocks = splitFileByBlockOfCode(file_text)
                    for block in blocks:
                        cleaned_text = cleanTextChunk(block)
                        important_terms = removeUnwantedTerms(wordTokenizer.tokenize(block))
                        data['file_name'].append(file)
                        data['text'].append(cleaned_text)
                        data['important_terms'].append(important_terms)
                except UnicodeDecodeError:
                    print(f"Could not read file {file}, skipping.")
                    continue
    dataset = Dataset.from_dict(data)
    return dataset

def init_dataset(path_folder: str, load_from_hf = True) -> Dataset:
    """
    Inicializa el dataset ya sea cargandolo desde HF o procesando los archivos localmente.
    Parametros:
        path_folder: Ruta a la carpeta con los archivos a procesar
        load_from_hf: Booleano que indica si cargar desde HF o procesar localmente
    Retorna:
        dataset: El dataset cargado o procesado
    """
    if not load_from_hf:
        dataset = loadLocalDataset(path_folder)
    else:
        dataset = load_dataset(dataset_hf_id)
    return dataset


def init_faiss_index(embeddingModel) -> faiss.IndexFlatL2:
    index = faiss.IndexFlatIP(embeddingModel.get_sentence_embedding_dimension())
    return index



def main(): #TODO: finish testing everything
    dataset = init_dataset(repo_base_path, load_from_hf=False)
    
    embedding_model = SentenceTransformer(embedding_model_name)
    
    faiss_index = init_faiss_index(embedding_model)
    
    embeddings = embedding_model.encode(dataset['important_terms'].tolist(), convert_to_numpy=True) # Here we avoid embeding all the text, only the important terms that define the usage of the code block.
    
    faiss_index.add(embeddings)
    
    print(f"Number of vectors in the index: {faiss_index.ntotal}")
    return dataset, faiss_index, embedding_model