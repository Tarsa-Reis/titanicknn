import pandas as pd
class KNN:
    def __init__(self, nearests = 5):
        self.nearests = 5

    def predicao(self,medidas:list,X:pd.DataFrame,y:pd.DataFrame,coordX:pd.DataFrame):
        
        knn = [[0]* 3 for _ in range(self.nearests)]
        j = 0
        while(j<self.nearests and j<X.shape[0]):
            soma = 0
            for i in range(1,len(medidas)):
                soma += (((X.iloc[j,i] - coordX.iloc[i]) ** 2) ** 0.5)

            knn[j][0] = X.iloc[j,0]
            knn[j][1] = soma
            knn[j][2] = y.iloc[j]
            j+=1
        knn.sort(key=lambda x: x[1],reverse=True)
            
        

        if(X.shape[0] > self.nearests):      
            for j in range(5,X.shape[0]):
                #print("fechado")
                soma = 0
                for i in range(1,len(medidas)):
                    #print("aberto")
                    soma += (((X.iloc[j,i] - coordX.iloc[i]) ** 2) ** 0.5)

                if(soma<knn[0][1]):
                    knn[0][0] = X.iloc[j,0]
                    knn[0][1] = soma
                    knn[0][2] = y.iloc[j]
                    knn.sort(key=lambda x: x[1],reverse=True)
                
        soma = 0
        for i in range (self.nearests):
            soma += knn[i][2]

        
        print(coordX.iloc[0])
        return (coordX.iloc[0], 1 if soma>=0.5 else 0)


def Normalizador(X:pd.DataFrame, medidas:list, max = None):
    a=0
    if(max==None):
        max=[0.0]*(len(medidas)-1)
        a=1
        for i in range(X.shape[0]):
            for j in range(1,len(medidas)):
                if(X.iloc[i,j]>max[j-1]):
                    max[j-1]=float(X.iloc[i,j])


    X = X.astype(float) 
    for j in range(1, len(medidas)):
        X.iloc[:,j] = X.iloc[:,j]/max[j-1]


    return X,max
        


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

#print(X_normalizado.dtypes)

knn = KNN(5)
data = {
    'PassengerId':[],'Survived':[]
}


for i in range(testData.shape[0]):
    id, predicao = knn.predicao(medidas,X_normalizado,y,testX_normalizado.iloc[i])
    data['PassengerId'].append((id).astype(int))
    data['Survived'].append(predicao)


df = pd.DataFrame(data)

df.to_csv('Resposta.csv',index=False)



