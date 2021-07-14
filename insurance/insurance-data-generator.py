import numpy as np
import pandas as pd
import numpy.random
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook as tqdm

def get_modelmonitoring_insurance_dataset():
    data = pd.read_csv('insurance.csv')
    train = data.drop(['charges'],axis=1)
    y = data['charges']
    a = np.arange(0, train.shape[1])
    train_aug = pd.DataFrame(index=train.index, columns=train.columns, dtype='float64')

    for i in tqdm(range(0, len(train))):
        AUG_FEATURE_RATIO = 0.5
        AUG_FEATURE_COUNT = np.floor(train.shape[1]*AUG_FEATURE_RATIO).astype('int16')

        aug_feature_index = np.random.choice(train.shape[1], AUG_FEATURE_COUNT, replace=False)
        aug_feature_index.sort()

        
        feature_index = np.where(np.logical_not(np.in1d(a, aug_feature_index)))[0]

        
        train_aug.iloc[i, feature_index] = train.iloc[i, feature_index]

      
        rand_row_index = np.random.choice(len(train), len(aug_feature_index), replace=True)

        
        for n, j in enumerate(aug_feature_index):
            train_aug.iloc[i, j] = train.iloc[rand_row_index[n], j]
    
    
    train_aug['charges'] = y
    train_dataset,predict_data = train_test_split(train_aug, test_size=0.1)
    train_dataset.to_csv('train.csv',index=False)
    
    GTpredict_data1 = predict_data.iloc[0:30,:]
    GTpredict_data2 = predict_data.iloc[30:60,:]
    GTpredict_data3 = predict_data.iloc[60:90,:]
    GTpredict_data4 = predict_data.iloc[90:120,:]
    GTpredict_data5 = predict_data.iloc[120:,:]
    
    GTpredict_data1.to_csv('GTpredict_data1.csv',index=False)
    GTpredict_data2.to_csv('GTpredict_data2.csv',index=False)
    GTpredict_data3.to_csv('GTpredict_data3.csv',index=False)
    GTpredict_data4.to_csv('GTpredict_data4.csv',index=False)
    GTpredict_data5.to_csv('GTpredict_data5.csv',index=False)
        
    
    predict_data1 = GTpredict_data1.iloc[0:30,:].drop(['charges'],axis=1)
    predict_data2 = GTpredict_data2.iloc[30:60,:].drop(['charges'],axis=1)
    predict_data3 = GTpredict_data3.iloc[60:90,:].drop(['charges'],axis=1)
    predict_data4 = GTpredict_data4.iloc[90:120,:].drop(['charges'],axis=1)
    predict_data5 = GTpredict_data5.iloc[120:,:].drop(['charges'],axis=1)
    
    predict_data1.to_csv('predict_data1.csv',index=False)
    predict_data2.to_csv('predict_data2.csv',index=False)
    predict_data3.to_csv('predict_data3.csv',index=False)
    predict_data4.to_csv('predict_data4.csv',index=False)
    predict_data5.to_csv('predict_data5.csv',index=False)
    
    
    drifted_data = GTpredict_data1.iloc[0:30,:].drop(['charges'],axis=1)
    for i in range(0,len(drifted_data)):
        if drifted_data['age'].iloc[i]>=30:
            drifted_data['age'].iloc[i]=drifted_data['age'].iloc[i]+30
        if drifted_data['bmi'].iloc[i]>=30:
            drifted_data['bmi'].iloc[i] = drifted_data['bmi'].iloc[i]+15
    drifted_data.to_csv('drifted_data1.csv',index=False)
    
    drifted_data2 = GTpredict_data2.iloc[0:30,:].drop(['charges'],axis=1)
    drifted_data2['sex']='male'
    drifted_data2['sex'].iloc[29]='female'
    drifted_data2['region']='northwest'
    drifted_data2['region'].iloc[28]='southeast'
    drifted_data2['region'].iloc[29]='southeast'
    drifted_data2.to_csv('drifted_data2.csv',index=False)
    
if __name__ == "__main__":
    get_modelmonitoring_insurance_dataset()
     
