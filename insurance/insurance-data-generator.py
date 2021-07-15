import numpy as np
import pandas as pd
import numpy.random,argparse,uuid
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from randomtimestamp import randomtimestamp
from sklearn import preprocessing as skpreprocessing
from datetime import datetime
from tqdm import tqdm_notebook as tqdm

def get_modelmonitoring_insurance_dataset(start,end):
    start_list = start.split('-')
    end_list = end.split('-')
    start = datetime(int(start_list[2]),int(start_list[1]),int(start_list[0]),int(start_list[3]),int(start_list[4]),int(start_list[5]))
    end = datetime(int(end_list[2]),int(end_list[1]),int(end_list[0]),int(end_list[3]),int(end_list[4]),int(end_list[5]))
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
    
    
    train_aug['charges'] = y+y*0.03
    train_all = pd.concat([data,train_aug])
  
    for col in ['sex', 'smoker', 'region']:
        if (train_all[col].dtype == 'object'):
            le = skpreprocessing.LabelEncoder()
            le = le.fit(train_all[col])
            train_all[col] = le.transform(train_all[col])
            print('Completed Label encoding on',col)
    
    train_dataset,predict_data = train_test_split(train_all, test_size=0.1)
    cols = train_dataset.columns.tolist()
    cols = cols[-2:]+cols[:-2]
    train_dataset = train_dataset[cols]
    train_dataset.to_csv('train.csv',index=False)
    
    insurance_input = train_dataset.drop(['charges'],axis=1)
    insurance_target = train_dataset['charges']
    x_scaled = StandardScaler().fit_transform(insurance_input)
    linReg = LinearRegression()
    linReg_model = linReg.fit(x_scaled, insurance_target)
    
    predict_data = predict_data.drop(['charges'],axis=1)
    predict_data['model_target'] = linReg.predict(predict_data)
    predict_data = predict_data.reset_index(drop=True)
    for i in range(len(predict_data)):
        predict_data.loc[i,'timestamp'] = randomtimestamp(start=start, end=end)
        predict_data.loc[i,'unique_id'] = uuid.uuid4()
    predict_data['GT_target'] = predict_data['model_target'] + predict_data['model_target']*0.05
    
    GTpredict_data1 = predict_data.iloc[0:30,:].drop(['model_target'],axis=1)
    GTpredict_data2 = predict_data.iloc[30:60,:].drop(['model_target'],axis=1)
    GTpredict_data3 = predict_data.iloc[60:90,:].drop(['model_target'],axis=1)
    GTpredict_data4 = predict_data.iloc[90:120,:].drop(['model_target'],axis=1)
    GTpredict_data5 = predict_data.iloc[120:151,:].drop(['model_target'],axis=1)
    
    GTpredict_data1.to_csv('GTpredict_data1.csv',index=False)
    GTpredict_data2.to_csv('GTpredict_data2.csv',index=False)
    GTpredict_data3.to_csv('GTpredict_data3.csv',index=False)
    GTpredict_data4.to_csv('GTpredict_data4.csv',index=False)
    GTpredict_data5.to_csv('GTpredict_data5.csv',index=False)
        
    
    predict_data1 = predict_data.iloc[0:30,:].drop(['GT_target'],axis=1)
    predict_data2 = predict_data.iloc[30:60,:].drop(['GT_target'],axis=1)
    predict_data3 = predict_data.iloc[60:90,:].drop(['GT_target'],axis=1)
    predict_data4 = predict_data.iloc[90:120,:].drop(['GT_target'],axis=1)
    predict_data5 = predict_data.iloc[120:151,:].drop(['GT_target'],axis=1)
    
    predict_data1.to_csv('predict_data1.csv',index=False)
    predict_data2.to_csv('predict_data2.csv',index=False)
    predict_data3.to_csv('predict_data3.csv',index=False)
    predict_data4.to_csv('predict_data4.csv',index=False)
    predict_data5.to_csv('predict_data5.csv',index=False)
    
    
    drifted_data = GTpredict_data1.iloc[0:30,:].drop(['GT_target'],axis=1)
    for i in range(0,len(drifted_data)):
        if drifted_data['age'].iloc[i]>=30:
            drifted_data['age'].iloc[i]=drifted_data['age'].iloc[i]+30
        if drifted_data['bmi'].iloc[i]>=30:
            drifted_data['bmi'].iloc[i] = drifted_data['bmi'].iloc[i]+15
    drifted_data.to_csv('drifted_data1.csv',index=False)
    
    drifted_data2 = GTpredict_data2.iloc[0:30,:].drop(['GT_target'],axis=1)
    drifted_data2['sex']='male'
    drifted_data2['sex'].iloc[29]='female'
    drifted_data2['region']='northwest'
    drifted_data2['region'].iloc[28]='southeast'
    drifted_data2['region'].iloc[29]='southeast'
    drifted_data2.to_csv('drifted_data2.csv',index=False)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", dest="start", default=None, type=str, help="start-timestamp")
    parser.add_argument("--end", dest="end", required=True, type=str, help="end-timestamp")

    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    start_timestamp = FLAGS.start
    end_timestamp = FLAGS.end
    get_modelmonitoring_insurance_dataset(start_timestamp,end_timestamp)
     
