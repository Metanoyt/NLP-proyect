import json
import binascii

DEBUG = False

class HexSaver:

    def saveState(path, state):
        '''
        Recibe un objeto el cual cambia a bytes, lo codifica en hexadecimal y lo guarda en un archivo
        
                Parametros:
                        path (String): ruta hacia la carpeta que contiene archivos
                        state (List): una lista de objetos
        '''
        with open(path, 'wb') as f:
            #strState = str(state)
            #binState = bytearray(strState, encoding = "utf-8")
            #binState = bytes(binState)
            binState = json.dumps(state).encode('utf-8')
            binState = binascii.hexlify(binState)
            f.write(binState)
            if DEBUG:
                print(type(binState))
            f.close()

    def loadState(path):
        '''
        Retorna una lista de objetos la cual es decodificada y transformada de bytes a objetos
        
                Parametros:
                        path (String): ruta hacia la carpeta que contiene archivos
                Retorna:
                        res_dict (List): Lista de objetos
        '''
        with open(path, 'rb') as f:
            binState = f.read()
            binState = binascii.unhexlify(binState)
            strState = binState.decode('utf-8')
            res_dict = json.loads(strState) 
            if DEBUG:
                print(res_dict)
            f.close()
            return res_dict

#lista = HexSaver.loadState('./search/SavedStates')
#print(lista[:0])