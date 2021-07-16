## Insurance data generator
1. insurance-data-generator.py generates data between the specified timestamps.
2. --n_predictdatasets signifies number of predict datasets to generate
3. -n_GTdatasets signifies number of Ground truth datasets to generate
4. -n_driftedatasets signifies number of drift datasets to generate
5. --start '07-08-2021-0-0-0' signifies the start timestamp eg (i.e 8 july 2021)
6. -duration signifies number of the days for which data to be generated, (eg 3, means starting from 8 july, it will generate data for 8, 9 and 10 July)
## Prerequisites
1. pip install sklearn
2. pip install tqdm
3. pip install randomtimestamp
## How to Run
4. Clone the repo and run using command 
5. `!python data-generator.py --n_predictdatasets 5 --n_GTdatasets 3 --n_driftedatasets 2 --start '07-08-2021-0-0-0' --duration 3`
6. It will generate train.csv file, and user specified number of datasets.
