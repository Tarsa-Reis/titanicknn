import pandas as pd
class KNN:
    def __init__(self, nearests = 5, p = 1):
        self.nearests = nearests
        self.p = p

    def euclidiana(self,j:int,X:pd.DataFrame,testeX:pd.DataFrame):
        soma =0
        for i in range(1,X.iloc[0].size):
                soma += ((X.iloc[j,i] - testeX.iloc[i]) ** 2) 
        soma = (soma ** 0.5)
        return soma
    
    def minkowski(self,j:int,X:pd.DataFrame,testeX:pd.DataFrame, p:int):
        soma =0
        for i in range(1,X.iloc[0].size):
            diferenca = (X.iloc[j, i]) - (testeX.iloc[i])
            if diferenca != 0 or p >= 0:
                soma += abs(diferenca) ** p
        if(soma!=0):
            soma = (soma ** (1/p))
        return soma
    
    def minkowskiZero(self,j:int,X:pd.DataFrame,testeX:pd.DataFrame,p:int):
        soma =0
        for i in range(1,X.iloc[0].size):
            if(X.iloc[j,i] != testeX.iloc[i]):
                soma += 1
        return soma
    

    def predicao(self,X:pd.DataFrame,y:pd.DataFrame,testeX:pd.DataFrame):

        if(self.p==0):
            funcao = self.minkowskiZero
        else:
            funcao = self.minkowski
        
        knn = [[0]* 3 for _ in range(self.nearests)]
        j = 0
        while(j<self.nearests and j<X.shape[0]):
            soma = funcao(j,X,testeX,(self.p))

            knn[j][0] = int(X.iloc[j,0])
            knn[j][1] = float(soma)
            knn[j][2] = int(y.iloc[j])
            j+=1
        knn.sort(key=lambda x: x[1],reverse=True)
            
        
     
        for j in range(self.nearests,X.shape[0]):
            soma = funcao(j,X,testeX,self.p)

            if(soma<knn[0][1]):
                knn[0][0] = int(X.iloc[j,0])
                knn[0][1] = float(soma)
                knn[0][2] = int(y.iloc[j])
                knn.sort(key=lambda x: x[1],reverse=True)
                
        survived = 0
        notsurvived = 0
    
        if 0.0 in [ n[1] for n in knn]:
            for i in range (self.nearests):
                if (knn[i][1]==0.0):
                    if(knn[i][2]==1):
                        survived += 1
                    else:
                        notsurvived += 1
        else:
            for i in range (self.nearests):
                if(knn[i][2]==1):
                    survived += (1/knn[i][1])
                else:
                    notsurvived += (1/knn[i][1])
        
        


        return (testeX.iloc[0], 1 if survived>notsurvived else 0)


def Normalizador(X:pd.DataFrame, medidas:list, max = None):
    a=0
    if(max==None):
        max=[0.0]*(X.iloc[0].size-1)
        a=1
        for i in range(X.shape[0]):
            for j in range(1,X.iloc[0].size):
                if(X.iloc[i,j]>max[j-1]):
                    max[j-1]=float(X.iloc[i,j])


    X = X.astype(float) 
    for j in range(1, len(medidas)):
        X.iloc[:,j] = X.iloc[:,j]/max[j-1]


    return X,max
        

def testa(y:pd.DataFrame,predicao:pd.DataFrame):
    df = pd.DataFrame()
    df['Acertos'] = y.reset_index(drop=True).iloc[:] == predicao.iloc[:,1]  # Compara

    # Calculando a porcentagem de acertos
    porcentagem_acertos = df['Acertos'].mean() * 100 

    return porcentagem_acertos


trainData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')

medidas = ['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
label = 'Survived'

X = trainData[medidas]
y = trainData[label]

testX = testData[medidas]

X_encoded = pd.get_dummies(X, columns=['Sex', 'Embarked'])
X_encoded = X_encoded.fillna(X_encoded.mean()).astype(int)


X_normalizado,max = Normalizador(X_encoded,medidas)


testX_encoded = pd.get_dummies(testX, columns=['Sex', 'Embarked'])
testX_encoded = testX_encoded.fillna(testX_encoded.mean()).astype(int)

testX_normalizado,max = Normalizador(testX_encoded,medidas,max)

#A seguinte parte foi para teste com os dados do próprio train e a descoberta do melhor numero de vizinhos, j
#  é o intervalo de vizinhos para testar, p é para o intervalo de teste do grau de distancia entre os pontos de minkowski(1 para manhatan e 2 para euclidiana):
#---------------------------------------------------------------------------------------------------------
tamanho = (X_normalizado.shape[0])

Xtrain = X_normalizado.iloc[:tamanho-int(tamanho/10)]
ytrain = y.iloc[:tamanho-int(tamanho/10)]
#print(Xtrain)

Xtest = X_normalizado.iloc[tamanho-int(tamanho/10):]

for p in range (-3,4):
    for j in range(3,9,2):
        knn = KNN(j,p)
        data = {'PassengerId':[],'Survived':[]}
        for i in range(Xtest.shape[0]):
            #print(Xtest.iloc[i])
            id, predicao = knn.predicao(Xtrain,y,Xtest.iloc[i])
            data['PassengerId'].append(int(id))
            data['Survived'].append(predicao)

        print(p,j,testa(y.iloc[tamanho-int(tamanho/10):],pd.DataFrame(data)))

        #df = pd.DataFrame(data)
        #df.to_csv('teste.csv',index=False)

#-------------------------------------------------------------------------------------------


'''knn = KNN(3,1)
data = {'PassengerId':[],'Survived':[]}
for i in range(testData.shape[0]):
    id, predicao = knn.predicao(X_normalizado,y,testX_normalizado.iloc[i])
    data['PassengerId'].append((id).astype(int))
    data['Survived'].append(predicao)


df = pd.DataFrame(data)

df.to_csv('knnManual3Minkowski1Pesos.csv',index=False)
'''



    






