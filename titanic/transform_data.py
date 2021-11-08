class Transformer():
    def preprocess(dataframe):
        data_to_preprocess = dataframe 
        data_to_preprocess = data_to_preprocess.fillna(data_to_preprocess.mean())
        features = ["Pclass", "Sex", "SibSp", "Parch"]
        train_df = pd.get_dummies(data_to_preprocess[features])
        data_to_preprocess = pd.concat([data_to_preprocess[["Age", "Fare", "Survived", "PassengerId"]], train_df], axis=1)
        data_to_preprocess = data_to_preprocess.drop(["PassengerId","Survived"], 1).values
        preprocessed_data = data_to_preprocess
        return preprocessed_data
