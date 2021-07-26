## Insurance data generator
1. insurance-data-generator.py generates data between the specified time.
2. --n_predictdatasets signifies number of predict datasets to generate
3. -n_GTdatasets signifies number of Ground truth datasets to generate
4. -n_driftedatasets signifies number of drift datasets to generate
5. --start '07-08-2021-0-0-0' signifies the start timestamp eg (i.e 8 july 2021). start parameter in year-month-date-hours-minutes-seconds format.
6. -duration signifies number of the hours,minutes and seconds (eg 10-20-1, means starting from 8 july, it will generate data for 10 hours,20minutes and 1 sec, from start ('07-08-2021-0-0-0').
7. **Note** : The number of GTdatasets and drifted datasets should be less than or equal to number of predict datasets.
## Prerequisites
1. pip install sklearn
2. pip install tqdm
3. pip install randomtimestamp
## How to Run
4. Clone the repo and run using command 
5. `!python insurance-data-generator.py --n_predictdatasets 5 --n_GTdatasets 3 --n_driftedatasets 2 --start '07-08-2021-0-0-0' --duration '10-20-16'
6. It will generate training-data.csv file, and user specified number of datasets.
