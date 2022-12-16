import cv2
import os
import glob
import random
import numpy as np
from PIL import Image
#(radius, vizinhos, gridX, GridY, Trashoud)
#lbph = cv2.face.LBPHFaceRecognizer_create(2, 2, 7, 7, 50)

def getImagemComIdArface():
    
    caminho = (glob.glob("./ARFACE/*.bmp"))

    if os.path.exists("arfaceSet.txt"):
        os.remove("arfaceSet.txt")
        file = open("arfaceSet.txt", "x")
        file = open("arfaceSet.txt", "a")
    else:
        file = open("arfaceSet.txt", "x")
        file = open("arfaceSet.txt", "a")

    faces = []
    ids = []

    for f in caminho:

        imagemFace = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(f)[-1].split('-')[1])
        gender = os.path.split(f)[-1].split('-')[0]
        day = int(os.path.split(f)[-1].split('-')[-1].split('_')[0])
        

        if gender == str('Cw'):
            id = int(id + 76)

        if int(day) <= 13:
            ids.append(id)
            faces.append(imagemFace)

    list_set = set(ids)
    unique_list = (list(list_set))
    for x in unique_list:
        file.write(str(x) + '\n')

    file.close()

    return np.array(ids), faces


def trainamentoArface(ids, faces):

    lbph = cv2.face.LBPHFaceRecognizer_create(2, 2, 7, 7, 50)

    print('Treinamento dataset ARfaces iniciado....')

    lbph.train(faces, ids)
    lbph.write('classificadorArfaces.yml')

    print('Treinamento Concluído com sucesso!')

    return

def detectorFacialArfaces ():


    detectorFace = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    reconhecedor = cv2.face.LBPHFaceRecognizer_create()
    reconhecedor.read("classificadorArfaces.yml")

        
    contador = 0
    confiancaGer = []

    totalAcertos = 0
    percentualAcerto = 0.0
    totalConfianca = 0.0

    caminhos = [os.path.join('datasets/Arface_mtcnn_v2/face', f) for f in os.listdir('datasets/Arface_mtcnn_v2/face')]

    print(len(caminhos))

    for caminhoImagem in caminhos:
        day = int(os.path.split(caminhoImagem)[-1].split('-')[-1].split('_')[0])

        if day >= 14:

            contador += 1

            imagemFace = Image.open(caminhoImagem).convert('L')
            imagemFaceNP = np.array(imagemFace, 'uint8')

            facesDetectadas = detectorFace.detectMultiScale(imagemFaceNP)
            for (x, y, l, a) in facesDetectadas:

                idPrevisto, confianca = reconhecedor.predict(imagemFaceNP)

                idAtual = int(os.path.split(caminhoImagem)[-1].split('-')[1])
                gender = os.path.split(caminhoImagem)[-1].split('-')[0]

                if gender == str('Cw'):
                    idAtual = int(idAtual + 76)


                #cv2.rectangle(imagemFaceNP, (x, y), (x + l, y + a), (0, 255, 255), 2)
                #cv2.imshow("Face", imagemFaceNP)

                if idPrevisto == idAtual:
                    totalAcertos += 1
                    totalConfianca += confianca

                #print(str(idAtual) + " foi classificado como " + str(idPrevisto) + " - " + str(confianca))

                confiancaGer.append(confianca)

                #cv2.waitKey(500)
    print(totalAcertos)
    percentualAcerto = (totalAcertos / contador) * 100
    totalConfianca = totalConfianca / totalAcertos
    print("Percentual de acerto: " + str(percentualAcerto))
    print("Confiança total: " + str(totalConfianca))

    return confiancaGer

def getImagemComIdFrgc():

    caminho = (glob.glob("./FRGC/*.JPG"))

    if os.path.exists("frgcSet.txt"):
        os.remove("frgcSet.txt")
        file = open("frgcSet.txt", "x")
        file = open("frgcSet.txt", "a")
    else:
        file = open("frgcSet.txt", "x")
        file = open("frgcSet.txt", "a")
    faces = []
    ids = []

    randomtest = random.sample(caminho, 40)

    for f in caminho:
        if f not in randomtest:
            imagemFace = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
            id = int(os.path.split(f)[-1].split('d')[0])
            ids.append(id)
            faces.append(imagemFace)

    list_set = set(ids)
    unique_list = (list(list_set))
    for x in unique_list:
        file.write(str(x) + '\n')

    file.close()

    return np.array(ids), faces, randomtest

def trainamentoFrgc(ids,faces):

    lbph = cv2.face.LBPHFaceRecognizer_create(4, 8, 8, 8, 200)

    print('Treinamento dataset FRGC iniciado....')

    lbph.train(faces, ids)
    lbph.write('classificadorFrgc.yml')

    print('Treinamento Concluído com sucesso!')

    return

def detectorFacialFRGC(randomtest):

    detectorFace = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    reconhecedor = cv2.face.LBPHFaceRecognizer_create()
    reconhecedor.read("classificadorFrgc.yml")

    contador = 0
    
    confiancaGer = []

    totalAcertos = 0
    percentualAcerto = 0.0
    totalConfianca = 0.0

    caminhos = (glob.glob("./datasets/FRGC/FRGC-2.0-dist/nd1/Fall2002/**/*"))

    print(len(caminhos))

    for caminhoImagem in caminhos:
        #print(caminhos)
        #print(randomtest)
        if caminhoImagem in randomtest:
            imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
            imagemFaceNP = np.array(imagemFace, 'uint8')

            contador += 1

            facesDetectadas = detectorFace.detectMultiScale(imagemFaceNP)
            for (x, y, l, a) in facesDetectadas:

                idPrevisto, confianca = reconhecedor.predict(imagemFaceNP)

                idAtual = int(os.path.split(caminhoImagem)[-1].split('d')[0])

                #cv2.rectangle(imagemFaceNP, (x, y), (x + l, y + a), (0, 255, 255), 2)
                #cv2.imshow("Face", imagemFaceNP)

                if idPrevisto == idAtual:
                    totalAcertos += 1
                    totalConfianca += confianca

                #print(str(idAtual) + " foi classificado como " + str(idPrevisto) + " - " + str(confianca))

                confiancaGer.append(confianca)

                #cv2.waitKey(500)
    print(totalAcertos)
    percentualAcerto = (totalAcertos / contador) * 100
    totalConfianca = totalConfianca / totalAcertos
    print("Percentual de acerto: " + str(percentualAcerto))
    print("Confiança total: " + str(totalConfianca))

    return confiancaGer
