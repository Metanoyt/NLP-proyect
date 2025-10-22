import os
from math import log
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer

class TfidfBuilder:
    
    PATH: str
    DEBUG: bool
    tfidf: list
    __idf: dict
    
    def __init__(self, path: str, debug = False, tfidf: list = None):
        self.PATH = path
        self.DEBUG = debug
        
        
        if tfidf == None:
            self.tfidf = []
            self.__idf = dict()
            self.__constructTFIDF()
            return
        tfidf = tfidf
    
    def __constructTFIDF(self):
        """
        Retorna una lista de las frecuencias inversas de los documentos (IDF's)

                Parametros:
                        path (String): Ruta hacia la carpeta que contiene la colección

                Retorna.
                        tfidf_list (List): Lista de IDF's de la colección
        """
        D, tf_list = self.__getTF()
        for d in tf_list:
            tfidf = dict()
            for term, tf in d[1].items():
                idf = self.__getIDF(term, tf_list, D)
                tfidf[term] = tf * idf
            self.tfidf.append([d[0], tfidf])
        if self.DEBUG:
            print("Terminado. Largo de tfidf_list: " + str(len(self.tfidf)))
    

    def __getTF(self):
        '''
        Retorna la cantidad de documentos y el calculo de frecuencia de termino (TF) de cada documento, en formato de lista.
        
                Parametros:
                        path (String): ruta hacia la carpeta que contiene archivos
                        
                Retorna:
                        D (int): cantidad de documentos
                        tf_list (List): Lista con el calculo de frecuencias para cada termino de la colección (TF's)
        '''
        if self.DEBUG:
            print("Obteniendo TF's")
        dir_list = os.listdir(self.PATH)
        doc_list = [(self.PATH + "/" + i) for i in dir_list if i.endswith(".txt")]
        tf_list = []
        for i in doc_list:
            tf_list.append(self.__getTFforFile(i))

        return [len(doc_list), tf_list]
    
    def __getIDF(self, term, tf_list, D):
        """
        Retorna el IDF de un termino

                Parametros:
                        term (String): Termino al que se obtendra el IDF
                        tf_list (List): Lista con el calculo de frecuencias para cada termino de la colección (TF's)
                        D (int): Cantidad de documentos en la colección

                Retorna:
                        idf (int): Frecuencia inversa de un termino en un documento (IDF aka Inversed Document Frecuency)
        """
        if term in self.__idf:
            return self.__idf[term]
        self.__calcIDF(term, tf_list, D)
        return self.__idf[term]


    def __calcIDF(self, term, tf_list, D):
        '''
        Calcula el IDF para un termino
        
                Parametros:
                        term (String): termino al que se calculara la frecuencia en la colección (IDF)
                        tf_list (List): Lista con el calculo de frecuencias para cada termino de la colección (TF's)
                        D (int): Cantidad de documentos en la colección
        '''
        if self.DEBUG:
            print("Calc IDF para " + term)
        df = 0
        for i in tf_list:
            if term in i[1]:
                df += 1
        self.__idf[term] = 1 + log(D / (df + 1))


    def __getTFforFile(self, path):
        """
        Funcion principal que recibe la ruta de un archivo y le aplica preprocesamiento y lo guarda 
        en un diccionario.
        Recibo la dirección del archivo 
        Retorno: Una lista de 2 elementos, el primer elemento es el nombre del archivo
                        el segundo elemento es el diccionario de terminos del archivo.
        """

        # Definir el diccionario
        dic = dict()
        # Abro el archivo de texto
        f = open(path, 'r', encoding="utf8")

        # GUardo texto en una variable
        texto = f.read()

        # Cierro el archivo
        f.close()

        # Realizo un preprocesamiento del texto
        tokens = self.__preprocesamiento(texto)

        # print(tokens[0])

        # Agrego los terminos y su frecuencia al diccionario
        i = 0
        lenTokens = len(tokens)
        while i != lenTokens:
            # Si la clave no existe se agrega
            if tokens[i] not in dic:
                new = {tokens[i]: 1}
                dic.update(new)
                i += 1
            else:
                dic[tokens[i]] += 1
                i += 1

        textName = path.split(sep='/')
        textName = textName[-1]
        # Retorno el nombre del archivo junto con su diccionario de terminos
        return [textName, dic]