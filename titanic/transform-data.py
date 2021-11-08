class Transformer():
    def preprocess(dataframe):
        data_to_preprocess = dataframe 
        data_to_preprocess = data_to_preprocess.drop(['Name','PassengerId','Cabin','Ticket','Embarked'],axis=1)
        data_to_preprocess["Age"].fillna(value=data_to_preprocess["Age"].median(), inplace=True)
        data_to_preprocess = data_to_preprocess[data_to_preprocess["Fare"] < 100]
        le = preprocessing.LabelEncoder()
        le = le.fit(data_to_preprocess['Sex'])
        data_to_preprocess['Sex'] = le.transform(data_to_preprocess['Sex'])
        print('Completed Label encoding on Sex column in the dataset')
        preprocessed_data = data_to_preprocess
        return preprocessed_data
