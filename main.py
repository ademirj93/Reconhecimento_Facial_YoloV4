# Importação de depêndências
import cv2
import time
#import treinamento as tr

#Códigos de execução do treinamento
#./darknet.exe detector train .\faceFrgc.data .\faceFrgc.cfg yolov4.conv.137
#./darknet.exe detector train .\faceFrgc.data .\faceFrgc.cfg .\bkp\faceFrgc_last.weights

#Comandos de cria;cão das listas de treinamento da rede
#tr.listFrgcCreate()
#tr.listArfaceCreate()
#tr.getImagemComIdArface()

dataset = "Frgc"
#dataset = "Arface"

# Cores das Classes
COLORS = [(0,255,255),(255,255,0),(0,255,0),(255,0,0)]

# Carragamento das classes
class_names = []
with open(f'face{dataset}.names',"r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

if dataset == "Frgc":
    inputFile = "Valid_FRGC"
    net = cv2.dnn.readNet(f'faceFrgc_final.weights',f'faceFrgc.cfg')
elif dataset == "Arface":
    inputFile = "Valid_ARFACE"
    net = cv2.dnn.readNet(f'faceArface_final.weights',f'faceArface.cfg')


# Abre um arquivo de para captura
cap = cv2.VideoCapture(f'{inputFile}.mp4')
x = 416

# Setando parâmetros da rede neural
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(x,x), scale=1/255)

# Coleta de frame e criação do obj de saida
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_size = (frame_width,frame_height)
fps = 20
output = cv2.VideoWriter(f'identify_{inputFile}.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, frame_size)

print(frame_size)

# Válida a abertura
if not cap.isOpened() :
    print("Falha na leitura da captura!")
    exit()

# Lendo os frames do video
while True:

    # Captura do frame
    ret, frame = cap.read()

    if not ret:
        print("Processo encerrado!")
        break

    # Começo da Contagem dos MS
    start = time.time()
    
    # Detecção (Frame, Trashout )
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)

    # Fim da Contagem
    end = time.time()

    # Percorrer todas as detecções
    for (classid, score, box) in zip(classes, scores, boxes):

        # Gerando uma cor para a classe
        color = COLORS[int(classid) % len(COLORS)]

        # Pegando o nome da classe pelo ID e o seu Score de Acuracia
        label = f"{class_names[classid]} : {score}"

        # Desenhando a Box de detecção
        cv2.rectangle(frame, box, color, 2)

        # Escrevendo o nome da Classe em cima da box do obj
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Calculando FPS
    fps_label = f"FPS: {round(1.0/(end - start)), 2}"

    # Escrevendo FPS na imagem
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Mostrar Imagem
    cv2.imshow("detections", frame)
    
    # Gravar o frame no output file
    output.write(frame)

    # Espera da Resposta
    if cv2.waitKey(1) == 27:
        break

# liberação da captura/camera e encerramento de todas as janelas
cap.release()
output.release()
cv2.destroyAllWindows()