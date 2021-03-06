# MODEL MONITORING INSURANCE EXAMPLE (UI)

## 1. Create Model Monitor
1. From model monitoring create a new monitor
2. Give a name.
3. Add tag d3qatest
4. select the added model.
5. Select Model Type: Regression
6. Change model run frequency to 5 hours. (in UI it’s five hours but it will run in every 5 mins because of d3qatest tag)
7. Submit

### 2. Add training data 
1. Name : insurance-training-data 
   - This is DKube local dataset, created by the training/retraining pipeline above

2. If training data source is S3/d3 select version: eg. v2 
   - Type : csv
3. If training data source is SQL
      - Add query: `select * from insurance` (the table name can be different)
4. Download the [transformer script](https://github.com/oneconvergence/dkube-examples/tree/monitoring/insurance/transform-data.py) and upload. 
5. Note: Download the script in your setup and then add it by browsing.
6. Save training data.

### 2. Upload train metrics
1. Dwonload the json from from [link](https://raw.githubusercontent.com/oneconvergence/dkube-examples/monitoring/insurance/train_metrics.json), and upload into train metrics tab.
2.  Click Save

### 3. Update Schema
1. Edit the model monitor
2. Go to schema and change
  - charges as prediction output.
  - unique_id as RowID
  - Timestamp as timestamp
3. Select all or interested Input features.
4. Click Next and save.

### 4. Configure Following Dataset in modelmonitor
**Predict Dataset**
1. If source S3 or local
  -  Dataset: {model-monitor}-predict
  -  Type: CSV
2. If source SQL
      - Dataset: insurance-data
      - Query: `select * from insurance_predict` (table will be added to the DB by the datagen script)

**Labelled Dataset**
1. If source S3 or local
  -  Dataset: {model-monitor}-groundtruth
  -  Type: CSV
2. If source SQL
      - Dataset: insurance-data
      - Query: `select * from insurance_gt` (table will be added to the DB by the datagen script)

- **Ground Truth Column Name**: GT_target
- **Prediction Column Name**: charges

### 5. Alerts
Add Feature Drift Alerts 
 - The datageneration script will be generating drift on the following features - age, sex, bmi, region. 
 - Suggest to configure a separate alert for each individual feature. 
 - Use a threshold between 0 to 1. generally advised 0.05 to 0.1 for all categorical or all continious columns columns,  0.05 to 0.01 for mixed categorical and continious columns columns.
 - It fires an alert when calculated drift goes under the configured threshold

Add Performance Decay Alerts
  - Create an alert and choose Perormance Decay from dropdown.
  - Selct percentage and choose metrics from down.
  - Provide percentage threshold value betweeen 5 to 10 and save.

### 6. Start Monitor.
Click on Start for the specific monitor on Modelmonitor dashboard. 
   - Modelmonitor can only be started in 'ready' state.
   - It can be stopped anytime. Previous data will not be erased.

### SMTP Settings
Configure your SMTP server settings on Operator screen. This is optional. If SMTP server is not configured, no email alerts will be generated.

## Data Generation
1. Open [data_generation.ipynb](https://github.com/pallavi-pannu-oc/model-monitoring/blob/SDK/insurance/data_generation.ipynb) notebook for generating predict and groundtruth datasets.
2. In 1st cell, Update Frequency according to what you set in Modelmonitor. If the d3qatest tag was provided replace it with to use frequency in minutes. For eg: for 5 minutes replace it with `5m` else use `5h` for hours assuming Frequency specified in monitor was 5.
3. Then Run All Cells. It will start Pushing the data, by default it will push the data to local.

## Retraining
1. Stop the modelmonitor.
3. Open resources.ipynb and set INPUT_TRAIN_TYPE = 'retraining' and run all the cells.
4. Open train.ipynb and run all the cells.
5. This creates a new version of dataset and a new version of model
   - New dataset version will be created for 'insurance-training-data' dataset
   - New model version will be created for 'insurance-model' model
6. Edit modelmonitor (UI)
   - Specify the new model version on basic page
   - Specify new dataset version on Training data page
   - Save & Submit
   - Click Next to go to the schema page and Accept the regenerated schema.
   - Wait for a few (30) sec
   - Start the modelmonitor

## Cleanup
1. After your expirement is complete, Open [resources.ipynb](https://github.com/pallavi-pannu-oc/model-monitoring/blob/SDK/insurance/resources.ipynb) and set CLEANUP=True in last Cleanup cell and run.
