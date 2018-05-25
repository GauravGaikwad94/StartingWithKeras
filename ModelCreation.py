from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import pandas as pd

def read_data():
    filename = 'dataset/train.csv'
    row_data = pd.read_csv(filename)
    ##Clean data : Remove empty cells
    data=row_data.dropna()
    
    ##convert string values to int
    data['A']=data['A'].astype('category')
    data['D']=data['D'].astype('category')
    data['E']=data['E'].astype('category')
    data['F']=data['F'].astype('category')
    data['G']=data['G'].astype('category')
    data['I']=data['I'].astype('category')
    data['J']=data['J'].astype('category')
    data['L']=data['L'].astype('category')
    data['M']=data['M'].astype('category')
    cat_columns = data.select_dtypes(['category']).columns
    data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)
    
    ##Input features
    X=data[data.columns[1:-1]]
    X_train=X.iloc[0:501,:]
    X_test=X.iloc[501:,:]
    ##Output column
    Y=data['P']
    Y_train=Y.iloc[0:501]
    Y_test=Y.iloc[501:]
    return X_train,Y_train,X_test,Y_test
def create_model(X_train,Y_train,X_test,Y_test):
    # create model
    model = Sequential()
    model.add(Dense(30, input_dim=15, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    
    # Fit the model
    model.fit(X_train,Y_train, epochs=2500, batch_size=50,validation_data=(X_test,Y_test))
    
    # evaluate the model
    scores = model.evaluate(X_test,Y_test)# -*- coding: utf-8 -*-
    print("Scores=",scores)
    
    result=model.predict(X_test, batch_size=1)
    print(result)
    result=result > 0.5
    print(result)
    model.save('conv1d_model.h5')
    return model



def load_saved_model():
    return load_model('conv1d_model.h5')

def predict():
    model=load_model('conv1d_model.h5')
    filename = 'dataset/test.csv'
    data = pd.read_csv(filename)
    ##Clean data : Remove empty cells
    #data=row_data.dropna()
    
    ##convert string values to int
    data['A']=data['A'].astype('category')
    data['D']=data['D'].astype('category')
    data['E']=data['E'].astype('category')
    data['F']=data['F'].astype('category')
    data['G']=data['G'].astype('category')
    data['I']=data['I'].astype('category')
    data['J']=data['J'].astype('category')
    data['L']=data['L'].astype('category')
    data['M']=data['M'].astype('category')
    cat_columns = data.select_dtypes(['category']).columns
    data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)
    X=data[data.columns[1:]]
    result=model.predict(X, batch_size=1)
    print(result)        
    res=[]
    for i in range(len(result)):
        if result[i] > 0.5: 
            res.append(1)
        else:
            res.append(0)
        
    print(res)
    df = pd.DataFrame({'id':data['id'],'P':res})
    #df['id']=data['id']
    
    #df=(pd.DataFrame({'P':res})
    #print(df)
    df.to_csv("result.csv",index=False)
#X_train,Y_train,X_test,Y_test=read_data()
#create_model(X_train,Y_train,X_test,Y_test)
predict()

