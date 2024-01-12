import os


from PIL import Image, ImageTk
from appJar import gui



class Regressore:

    #type_n dove
    # type è la normalizazzione
    # n è il tipo di grafico

    def __init__(self,nome,metrics,none_1=None,none_2=None,minmax_1=None,minmax_2=None,z_score_1=None,z_score_2=None):
        self.nome=nome
        self.none_1=none_1
        self.none_2=none_2
        self.minmax_1=minmax_1
        self.minmax_2=minmax_2
        self.z_score_1=z_score_1
        self.z_score_2=z_score_2
        self.metrics=metrics

class Scanner:



    def __init__(self,folder):
        self.toFind = ["LinearRegression", "Ridge", "SGDRegressor", "LARS", "LassoLars",
                  "BayesianRidge", "ARDRegression", "TweedieRegressor",
                  "DecisionTreeRegressor", "RandomForestRegressor", "KNeighborsRegressor",
                  "RadiusNeighborsRegressor", "GaussianProcessRegressor", "SVR", "NuSVR", "LinearSVR"]
        self.folder = folder

    def scann(self):
        lista=[]
        if os.path.exists(self.folder):
            listaRegressori=os.listdir(self.folder)
            print(listaRegressori)
            for regressore in listaRegressori:
                if regressore.startswith(".") or regressore.startswith("TabelleDiCorrelazione"):
                    listaRegressori.remove(regressore)
            print(listaRegressori)
            if listaRegressori.sort()==self.toFind.sort():
                for name in listaRegressori:
                    print(name)
                    path=self.folder+"/"+name+"/"
                    if os.path.exists(path+"None") and os.path.exists(path+"MinMaxScaler") and os.path.exists(path+"StandardScaler"):
                        metrics = dict()
                        Normalizzazioni=["None","MinMaxScaler","StandardScaler"]
                        for normalizazione in Normalizzazioni:
                            metrics[normalizazione]=dict()
                            try:
                                file= open(path+normalizazione+"/reportMetrics.txt","r")
                                flag,flag1=False,False
                                for line in file:
                                    ##print(line)
                                    if line.__contains__("Metrics - Kcross validation - :"):
                                        flag=True
                                        continue
                                    if flag:
                                        #print(line)
                                        if line.__contains__("MAE test mean:"):
                                            flag1=True
                                        if flag1:
                                            nome,value=line.replace("\n","").replace("    ","").replace(" test mean","").split(":")
                                            metrics[normalizazione][nome]=round(float(value),2)
                            except FileNotFoundError:
                                metrics[normalizazione]="Error"

                        try:
                            none_1 = Image.open(path + "None/distribuzioneErroreResiduo.png")
                        except FileNotFoundError :
                            none_1="Errore"
                        try:
                            none_2 = Image.open(path + "None/varianzaErroreResiduo.png")
                        except FileNotFoundError :
                            none_2="Errore"
                        try:
                            minmax_1 = Image.open(path + "MinMaxScaler/distribuzioneErroreResiduo.png")
                        except FileNotFoundError :
                            minmax_1="Errore"
                        try:
                            minmax_2 =  Image.open(path + "MinMaxScaler/varianzaErroreResiduo.png")
                        except FileNotFoundError :
                            minmax_2="Errore"
                        try:
                            z_score_1 = Image.open(path + "StandardScaler/distribuzioneErroreResiduo.png")
                        except FileNotFoundError :
                            z_score_1="Errore"
                        try:
                            z_score_2 = Image.open(path + "StandardScaler/varianzaErroreResiduo.png")
                        except FileNotFoundError :
                            z_score_2="Errore"
                        reg= Regressore(name, metrics,none_1, none_2, minmax_1, minmax_2, z_score_1, z_score_2)
                        lista.append(reg)
        return lista

x=Scanner("./analysis")
lista=x.scann()

def retuntString(disct:dict):
    stringa=""
    for key in disct.keys():
        stringa+=key+": "+str(disct[key])+"\n"
    return stringa
def update(value):
    print(app.listbox("list")[0])
    regressore=None
    for regressori in lista:
        if regressori.nome== app.listbox("list")[0]:
            regressore=regressori
    Errore=ImageTk.PhotoImage(Image.open("./defaultImage/notFound.png").resize((100,100)))
    flag1,flag2=False,False
    if regressore.none_1 is "Errore":
        app.setImageData("display1",Errore,fmt="PhotoImage")
        flag1=True
    else:
        app.setImageData("display1",ImageTk.PhotoImage(regressore.none_1.resize((100, 100))),fmt="PhotoImage")
        app.setImageSubmitFunction("display1", lambda: press("display1", regressore.none_1))
    if regressore.none_2 is "Errore":
        flag2=True
        app.setImageData("display2",Errore,fmt="PhotoImage")
    else:
        app.setImageData("display2",ImageTk.PhotoImage(regressore.none_2.resize((100, 100))),fmt="PhotoImage")
        app.setImageSubmitFunction("display2", lambda: press("display2", regressore.none_2))
    if flag1 or flag2:
       app.setLabel("Metriche1", "Errore")
    else:
        app.setLabel("Metriche1", retuntString(regressore.metrics["None"]))

    flag1,flag2=False,False
    if regressore.minmax_1 is "Errore":
        flag1=True
        app.setImageData("display3",Errore,fmt="PhotoImage")
    else:
        app.setImageData("display3",ImageTk.PhotoImage(regressore.minmax_1.resize((100, 100))),fmt="PhotoImage")
        app.setImageSubmitFunction("display3", lambda: press("display3", regressore.minmax_1))
    if regressore.minmax_2 is "Errore":
        flag2=True
        app.setImageData("display4",Errore,fmt="PhotoImage")
    else:
        app.setImageData("display4", ImageTk.PhotoImage(regressore.minmax_2.resize((100, 100))), fmt="PhotoImage")
        app.setImageSubmitFunction("display4", lambda: press("display4", regressore.minmax_2))
    if flag1 or flag2:
       app.setLabel("Metriche2", "Errore")
    else:
        app.setLabel("Metriche2", retuntString(regressore.metrics["MinMaxScaler"]))

    flag1,flag2=False,False
    if regressore.z_score_1 is "Errore":
        flag1=True
        app.setImageData("display5",Errore,fmt="PhotoImage")
    else:
        app.setImageData("display5",ImageTk.PhotoImage(regressore.z_score_1.resize((100, 100))),fmt="PhotoImage")
        app.setImageSubmitFunction("display5", lambda: press("display5", regressore.z_score_1))
    if regressore.z_score_2 is "Errore":
        flag2=True
        app.setImageData("display6",Errore,fmt="PhotoImage")
    else:
        app.setImageData("display6",ImageTk.PhotoImage(regressore.z_score_2.resize((100, 100))),fmt="PhotoImage")
        app.setImageSubmitFunction("display6", lambda: press("display6", regressore.z_score_2))
    if flag1 or flag2:
       app.setLabel("Metriche2", "Errore")
    else:
        app.setLabel("Metriche3", retuntString(regressore.metrics["StandardScaler"]))


def press(btn,regressor):
    app.setImageData("zoom_image",ImageTk.PhotoImage(regressor),fmt="PhotoImage")
    app.showSubWindow("ZoomIn - Grafico")


app = gui()
app.setTitle("RegressorComparator -GUI")
if len(lista) == 0:
    app.errorBox("RegressorComparator -GUI", "Qualcosa è andato storto! Prova ad eseguire main.py", parent=None)
else:
    app.setSize("550x400")
    app.setResizable(canResize=False)
    l= [regressore.nome for regressore in lista]

    # creiamo la finestra di zoom
    app.startSubWindow("ZoomIn - Grafico", modal=True)
    app.addImageData("zoom_image",ImageTk.PhotoImage(lista[0].none_1),fmt="PhotoImage")
    app.stopSubWindow()

    # start initial pane
    app.startPanedFrame("p1")
    app.listbox("list", l,row=len(lista),selected=0, submit=update,height=520,width=20)

    # start second, vertical pane inside initial pane
    app.startPanedFrameVertical("p2")
    app.addLabel("Nome1","Senza Normalizzazione:",column=0,colspan=3,row=0)
    app.addImageData("display1",ImageTk.PhotoImage(lista[0].none_1.resize((100, 100))),fmt="PhotoImage",row=1,column=0)
    app.setImageSubmitFunction("display1",lambda :press("display1",lista[0].none_1))
    app.addImageData("display2",ImageTk.PhotoImage(lista[0].none_2.resize((100, 100))),fmt="PhotoImage",row=1,column=1)
    app.setImageSubmitFunction("display2",lambda :press("display2",lista[0].none_2))
    app.addLabel("Metriche1", retuntString(lista[0].metrics["None"]),row=1,column=2)

    # start additional panes inside second pane
    app.startPanedFrame("p3")
    app.addLabel("Nome2", "Normalizzazione MinMax:", column=0, colspan=3, row=0)
    app.addImageData("display3",ImageTk.PhotoImage(lista[0].minmax_1.resize((100, 100))),fmt="PhotoImage",row=1,column=0)
    app.setImageSubmitFunction("display3",lambda :press("display3",lista[0].minmax_1))
    app.addImageData("display4",ImageTk.PhotoImage(lista[0].minmax_2.resize((100, 100))),fmt="PhotoImage",row=1,column=1)
    app.setImageSubmitFunction("display4",lambda :press("display4",lista[0].minmax_2))
    app.addLabel("Metriche2", retuntString(lista[0].metrics["MinMaxScaler"]),row=1,column=2)
    app.stopPanedFrame()

    # start additional panes inside second pane
    app.startPanedFrame("p4")
    app.addLabel("Nome3", "Normalizzazione z-score:", column=0, colspan=3, row=0)
    app.addImageData("display5",ImageTk.PhotoImage(lista[0].z_score_1.resize((100, 100))),fmt="PhotoImage",row=1,column=0)
    app.setImageSubmitFunction("display5",lambda :press("display5",lista[0].z_score_1))
    app.addImageData("display6",ImageTk.PhotoImage(lista[0].z_score_2.resize((100, 100))),fmt="PhotoImage",row=1,column=1)
    app.setImageSubmitFunction("display6",lambda :press("display6",lista[0].z_score_2))
    app.addLabel("Metriche3", retuntString(lista[0].metrics["StandardScaler"]),row=1,column=2)
    app.stopPanedFrame()


    # stop second & initial panes
    app.stopPanedFrame()
    app.stopPanedFrame()

app.go()
