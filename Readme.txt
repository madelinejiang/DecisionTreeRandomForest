
To run DTreeM.,py, you need to have pandas v25.0 or higher version, 
pathname prefix to the directory of datasets you will use
For Linux Server csgrads1
pip install pandas --user
pip install sklearn --user

Optional arguments:
-p Pathprefix; default=All_data/hw1_data

Required arguments:
-c clauses {300,500,100,1500,1800}
-d examples {100,1000,5000}
-e heuristic 0 for entropy 1 for variance (gini if RandomForest is chosen)
-o option: 1 for no pruning 1. for Reduced error pruning 2. Depth-based pruning
3. for Random Forest
-t print tree : 0 for no printing 1. to print trees created by algorithm

To run script:
USAGE EXAMPLE:

python DTreeMJ.py -c 300 -d 100 -e 0 -o 1 -t 0

Expected Output: 
Entropy Heuristic
pre prune accuracy vs validation data 0.595
pre prune accuracy vs test data 0.575
RE pruning
post prune accuracy vs validation data 0.66
post prune accuracy vs test data 0.59

python DTreeMJ.py -p /home/011/m/mx/mxj121230/hw1_data/all_data/ -c 300 -d 100 -e 0 -o 3 -t 0

Expected Output:
Begin Grid Search
{'max_features': 'auto', 'n_estimators': 1000, 'max_depth': 10}
Accuracy for test data on Random Forest using GridsearchCV:  0.875


