import os
import shutil
import re
# Try to import ToktokTokenizer from NLTK using known locations; if unavailable, provide a small fallback.
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

from HexSaver import HexSaver
from TfidfBuilder  import TfidfBuilder
from VectorModelSearch import VectorModelSearch


repo_base_path = "./sample_projects"
processing_base_path = "./pyfiles/"
original_files_path = "./original_files/"

tokenizer = ToktokTokenizer()
def preprocess_text(file_txt:str, tokenizer=tokenizer):
    """
    Preprocess the document by tokenizing and removing unwanted terms.
    """
    

    txt_no_special = re.sub(r"""[ 0-9 !?',".-<>(){}!\]:@%;&*/[/]""", " ", file_txt)
    tokens = tokenizer.tokenize(txt_no_special)# Potentially can skip tokenization here

    no_useless_terms = remove_unwanted_terms(tokens) 
    preprocessed_text = ' '.join(no_useless_terms)

    return preprocessed_text


def remove_unwanted_terms(tokens):
        """
        Función para remover las stop words de un set de tokens
        Entradas: Un texto o cadena de carácteres
        Salida: Una lista de tokens
        """
        useless_terms = ['def', '#', 'return', 'if', 'for', 'in', 'and', 'or', 'not', 'is', 'to', 'the', 'a', 'an', 'of', 'on', 'with', 'as', 'by', 'this', 'that', 'it', 'from', 'at', 'be', 'are', 'was', 'were', 'but', 'else', 'elif', 'import', 'class', 'self']
        
        filtered_tokens = [token for token in tokens if token not in useless_terms]
        return filtered_tokens

# Initialize the TFIDF Builder
def move_files_and_process():
    """
    Move all .py files from the repository to the processing directory after preprocessing their text.
    """

    list_of_files = []

    if not os.path.exists(processing_base_path):
        os.makedirs(processing_base_path)

    for root, dirs, files in os.walk(repo_base_path):
        for file in files:
            if file.endswith(".py"):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        file_text = f.read()
                except UnicodeDecodeError:
                    print(f"Could not read file {file}, skipping.")
                    continue
                except Exception as e:
                    print(f"An error occurred while reading file {file}: {e}")
                    continue
                preprocessed_text = preprocess_text(file_text)
                list_of_files.append(os.path.join(processing_base_path, file))
                # Save processed file to processing directory
                with open(os.path.join(processing_base_path, file), 'w', encoding='utf-8') as f:
                    f.write(preprocessed_text)
                # Copy original file to original_files directory
                if not os.path.exists(original_files_path):
                    os.makedirs(original_files_path)
                shutil.copy2(os.path.join(root, file), os.path.join(original_files_path, file))
                
    return list_of_files

def gather_and_tokenize_original_selection(selected_files, tokenizer):
    file_texts = {}
    for file_path in selected_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_text = f.read()
        except UnicodeDecodeError:
            print(f"Could not read file {file_path}, skipping.")
            continue
        except Exception as e:
            print(f"An error occurred while reading file {file_path}: {e}")
            continue
        tokens = tokenizer.tokenize(file_text)
        file_texts[file_path] = tokens

    return file_texts

def main():

    # Get all .py files in the repo and move them to processing directory after preprocessing
    # docs = move_files_and_process() # Note: not needed for TFIDF building, once preprocessing is done, we can build directly

    #tfidf_builder = TfidfBuilder(processing_base_path)
    #print("TF-IDF model built successfully.")
    #print(f"Number of items in state: {len(tfidf_builder.get_tfidf())}")

    #TODO: Super importante, necesitamos pasar el documento original para el contexto, no el preprocesado
    #display
    #for doc in docs:
        #print(f"TF-IDF for document: {doc}")
        #tfidf_vector = tfidf_builder.getTFforFile(doc)
        #print(tfidf_vector[:10])
        #for term, score in tfidf_vector[:10]:  # Display top 10 terms
        #    print(f"Term: {term}, TF-IDF: {score}")
        #print("\n")

    # Save the TFIDF model to disk
    # tfid_state = tfidf_builder.get_tfidf()
    
    # Get execution path
    tfidf_save_path = "indexing/tfidf/state/tfidf_model_state.hex"
    # HexSaver.saveState(tfidf_save_path, tfid_state)
    
    # print(f"TF-IDF model state saved to {tfidf_save_path}")


    searcher = VectorModelSearch(
        savePath=tfidf_save_path,
        colectionFolder=processing_base_path,
        resultCant=5
    )

    query = "ctx grad_output tensor_meta dtensor_spec.placements"
    results = searcher.search(query)
    selected_files = []
    print(f"Search results for query: '{query}'")
    for doc in results:
        original_file_path = os.path.join(original_files_path, doc)
        selected_files.append(original_file_path)
        print(f"Document: {original_file_path}")
    
    # Get original files and tokenize them
    file_texts = gather_and_tokenize_original_selection(selected_files, tokenizer)
    for file_path, tokens in file_texts.items():
        print(f"\nOriginal content tokens for file: {file_path}")
        print(tokens[:5])  # Print first 50 tokens for brevity

    

if __name__ == "__main__":
    main()