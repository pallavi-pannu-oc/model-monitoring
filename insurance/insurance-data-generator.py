import numpy as np
import pandas as pd
import numpy.random,argparse,uuid
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from randomtimestamp import randomtimestamp
from sklearn import preprocessing as skpreprocessing
import datetime
from tqdm import tqdm_notebook as tqdm


def save_dataset(data,name):
    file_name = name +'.csv'
    data.to_csv(file_name,index=False)


def get_modelmonitoring_insurance_dataset(n_predictdatasets,n_GTdatasets,n_driftedatasets,start_timestamp,duration):
    start_list = start_timestamp.split('-')
    start = datetime.datetime(int(start_list[2]),int(start_list[0]),int(start_list[1]),0,0,0)
    print("generating datasets from")
    print("start-timestamp",start)
    date_1 = datetime.datetime.strptime(start_timestamp, "%m-%d-%Y-%H-%M-%S")
    duration = duration.split('-')
    if(len(duration)<2):
        duration.append('0')
        duration.append('0')
    elif(len(duration)<3):
        duration.append('0')
        
    end = date_1 + datetime.timedelta(hours = int(duration[0]) , minutes = int(duration[1]), seconds = int(duration[2]))
    print("end-timestamp",end)
    
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
    train_dataset,predict_data = train_test_split(train_all, test_size=0.1)
    save_dataset(train_dataset,'training-data')
    
    for dataframe in [train_dataset,predict_data]:
        for col in ['sex', 'smoker', 'region']:
            if (dataframe[col].dtype == 'object'):
                le = skpreprocessing.LabelEncoder()
                le = le.fit(dataframe[col])
                dataframe[col] = le.transform(dataframe[col])
                print('Completed Label encoding on',col)
    
 
    insurance_input = train_dataset.drop(['charges'],axis=1)
    insurance_target = train_dataset['charges']
    x_scaled = StandardScaler().fit_transform(insurance_input)
    linReg = LinearRegression()
    linReg_model = linReg.fit(x_scaled, insurance_target)
    
    predict_data = predict_data.drop(['charges'],axis=1)

    predict_data['charges'] = linReg.predict(predict_data)
    predict_data = predict_data.reset_index(drop=True)
    for i in range(0,len(predict_data)):
        predict_data.loc[i,'timestamp'] = randomtimestamp(start=start, end=end)
        predict_data.loc[i,'unique_id'] = uuid.uuid4()
    
    predict_data = predict_data.sort_values(by=['timestamp'])
    n_predict_rows = int(predict_data.shape[0]/n_predictdatasets)
    index = 0
    for i in range(1,n_predictdatasets+1):
        pred_data = predict_data.iloc[index:index+n_predict_rows,:]
        pred_data_name = str(i)+'_predict_data'
        save_dataset(pred_data,pred_data_name)
        index += n_predict_rows
    
    print("predict datasets generation completed")  
    
    for i in range(1,n_GTdatasets+1):
        filename = str(i)+'_predict_data.csv'
        gt_data = pd.read_csv(filename)
        gt_data['GT_target'] = gt_data['charges'] + gt_data['charges']*0.05
        gt_data = gt_data.drop(['charges'],axis=1)
        gt_name = str(i)+'_GTpredict_data'
        save_dataset(gt_data,gt_name)
        
    print("GT datasets generation completed")  
    
    for j in range(1,n_driftedatasets+1):
        filename = str(j)+'_predict_data.csv'
        drifted_data = pd.read_csv(filename)
        if j%2==0:
            for i in range(0,len(drifted_data)):
                if drifted_data['age'].iloc[i]>=30:
                    drifted_data['age'].iloc[i]=drifted_data['age'].iloc[i]+30
                if drifted_data['bmi'].iloc[i]>=30:
                    drifted_data['bmi'].iloc[i] = drifted_data['bmi'].iloc[i]+15
        else:   
            drifted_data['sex']='male'
            drifted_data['sex'].iloc[29]='female'
            drifted_data['region']='northwest'
            drifted_data['region'].iloc[28]='southeast'
            drifted_data['region'].iloc[29]='southeast'
        drifted_data = drifted_data.drop(['charges'],axis=1)
        drifted_name = str(j)+'_drifted_data'
        save_dataset(drifted_data,drifted_name)
    print("drift datasets generation completed")  
     
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_predictdatasets", dest="n_predictdatasets", required=True, type=int, help="number of predict datasets to be generated")
    parser.add_argument("--n_GTdatasets", dest="n_GTdatasets", required=True, type=int, help="number of Ground Truth datasets to be generated")
    parser.add_argument("--n_driftedatasets", dest="n_driftedatasets", required=True, type=int, help="number of drifted datasets to be generated")
    parser.add_argument("--start", dest="start", default=None, type=str, help="start-timestamp")
    parser.add_argument("--duration", dest="duration", required=True, type=str, help="duration like 1 day, 2 day")

    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    n_predictdatasets = FLAGS.n_predictdatasets
    n_GTdatasets = FLAGS.n_GTdatasets
    n_driftedatasets = FLAGS.n_driftedatasets
    start_timestamp = FLAGS.start
    duration = FLAGS.duration
    get_modelmonitoring_insurance_dataset(n_predictdatasets,n_GTdatasets,n_driftedatasets,start_timestamp,duration)
     
