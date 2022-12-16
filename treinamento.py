import glob, os, random

def listFrgcCreate():

    train = open("trainfrgc.txt", "x")
    train = open("trainfrgc.txt", "a")

    test = open("testfrgc.txt", "x")
    test = open("testfrgc.txt", "a")

    caminho = (glob.glob("./FRGC/*.txt"))
    randomValue = random.sample(caminho, 48)

    for f in caminho:
        
        if f not in randomValue:
            train.write(f + '\n')
        else:
            test.write(f + '\n')

    return

def listArfaceCreate():

    train = open("trainarface.txt", "x")
    train = open("trainarface.txt", "a")

    test = open("testarface.txt", "x")
    test = open("testarface.txt", "a")

    caminho = (glob.glob("./ARFACE/*.txt"))

    for f in caminho:

        day = int(os.path.split(f)[-1].split('-')[-1].split('_')[0])
        
        if int(day) <= 13:
            train.write(f + '\n')
        else:
            test.write(f + '\n')

    return

def getImagemComIdArface():
    
    caminho = (glob.glob("./ARFACE/*.bmp"))

    ids = []

    id = 0

    for f in caminho:

        id = int(os.path.split(f)[-1].split('-')[1])
        gender = os.path.split(f)[-1].split('-')[0]
        file = os.path.split(f)[-1].split('.')[0]
        id = id - 1

        if gender == str('Cw'):
            id = int(id + 9)

        os.remove(f'./ARFACE/{file}.txt')
        doc = open(f'./ARFACE/{file}.txt', "x")
        doc = open(f'./ARFACE/{file}.txt', "a")

        doc.write(str(id) + " 0.502092 0.501515 0.995816 0.996970")

    doc.close()

    return