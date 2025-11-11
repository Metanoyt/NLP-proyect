import math
import re
import nltk
from os.path import exists
from HexSaver import HexSaver
from TfidfBuilder  import TfidfBuilder
#SAVE_PATH = './search/SavedStates'
#COLECTION_FOLDER = './search/archivos'
#RESULT_CANT = 5

class VectorModelSearch:
    
    tfidf: list
    RESULT_CANT: int
    COLECTION_FOLDER: str
    SAVE_PATH: str
    
    # Funcion para la inicializacion de los datos y configuracion del programa
    def __init__(self, savePath, colectionFolder, resultCant):
        self.SAVE_PATH = savePath
        self.COLECTION_FOLDER = colectionFolder
        self.RESULT_CANT = resultCant
        
        self.stopword_list = nltk.corpus.stopwords.words('english')
        self.ps = nltk.porter.PorterStemmer()
        self.tfidf = []
        
        if not exists(self.SAVE_PATH):
            self.tfidfBuilder = TfidfBuilder(self.COLECTION_FOLDER)
            self.tfidf = self.tfidfBuilder.tfidf
            HexSaver.saveState(self.SAVE_PATH, self.tfidf)
        self.tfidf = HexSaver.loadState(self.SAVE_PATH)
        self.tfidfBuilder = TfidfBuilder(self.COLECTION_FOLDER, tfidf=self.tfidf)
    
    def search(self, query: str):
        """
            Funcion para tratar cada busqueda
            Entrada: el texto de la conculta
            Comportamiento: tokeniza y filtra el texto para convertirlo en un vector con el cual realizar la busqueda
                            e imprimir el resultado
        """
        query = query.lower()
        query = re.sub(r"""[ 0-9 !?',".-<>(){}!\]:@%;&*/[/]""", " ", query)
        vector = {}
        for word in query.split(" "):
            if word not in self.stopword_list:
                root_word = self.ps.stem(word)
                if root_word in vector:
                    vector[root_word] += 1
                else:
                    vector[root_word] = 1
        # Busqueda
        results = self.__handle_query(vector)
        return results
    
    def __handle_query(self, query):
        """
            Obtiene los n documentos mas relevantes.

            Parametros: 
                arrayDocuments(Array de documentos): Es un array que contiene listas de documentos, los documentos son una lista con el nombre
                del documento y un diccionario con los calculos tfidf del documento.
                query(Array): Es un arreglo que contiene las frecuencias de los terminos.
                n(INT): Es un entero que define cuantos documentos relevantes se deben retornar.  
            
            Retorna: Retorna los nombres de los n documentos mas relevantes.  
        """
        dicAngle = {}
        for Document in self.tfidf:
            dicAngle[Document[0]] = self.__calculateAngles(Document[1], query)

        return self.__getNMax(dicAngle, self.RESULT_CANT)

    def __calculateAngles(self, tfidf: list, query):
        """
            calcula el coceno de un angulo de dos vectores.

            Parametros: 
                tfidf(Arreglo de INT): Es un vector que contiene los calculos tf-idf de un documento
                query(Arreglo de INT): Es un vector que contiene las frecuecias de los terminos de una consulta dada.
            
            Retorna: Retorna el coceno del angulo de los dos vectores de entrada.  
        """
        multiplicationSum = 0
        documentSum = 0
        querySum = 0
        term_list = query.keys()
        cos_angle = 0

        # Este ciclo calcula la sumatorias de cado uno de los factores de la formula
        for term in term_list:
            if term not in tfidf:
                querySum += query[term] ** 2
            else:
                multiplicationSum += (tfidf[term] * query[term])
                documentSum += tfidf[term] ** 2
                querySum += query[term] ** 2

        if documentSum > 0:
            if len(query) == 1:
                cos_angle = multiplicationSum / len(tfidf)
                # cosine = ( V1 * V2) / V2_n
            else:
                cos_angle = multiplicationSum / (math.sqrt(querySum) * math.sqrt(documentSum))
                # cosine  = ( V1 * V2 ) / ||V1|| x ||V2||}
                # cosine = documentVector * queryVector / ||documentVector|| x ||queryVector||

        return cos_angle

    def __getNMax(self, dic, n: int):
        """
            Obtiene los n nombres de documentos con mayor valor.

            Parametros: 
                dic(Diccionario): Es un diccionario cuyo donde la llave es el nombre del documento y el valor es el coceno del angulo.
                n(INT): Es un entero que especifica la cantidad de nombres de documentos a retornar.
            
            Retorna: Retorna las n llaves con mayor valor.  
        """
        result = []
        for i in range(0, n):
            key = max(dic, key=dic.get)
            result.append(key)
            dic.pop(key)
        return result