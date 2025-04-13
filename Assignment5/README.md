# Assignment 3

## Team 3

We are using the libraries described on the Assignment 1 file:
* scikit-learn 1.6.1
* matlplotlib 3.10.0
* pandas 2.2.3
* fastai 2.8.0
* imamgecodecs 2024.12.30

No other library was used for this assignment.

## Files included in the zip
The following files are included in the response:

Assignment's question answers:
* *question1.py*
* *question2.py*
* *question3.py*
* *question4.py*
* *report.pdf*
* This *README.md* file

Additional Files:
* *utils.py*

## Assignment's question answers

**Note:** *For all of this to work properly, 'datasets' folder **should be accessible** (in the same directory) as the python files.*

To run each of the answers (1 - 3), just call:
```
python .\question1.py
```

Questions 1 to 3 will plot something, question 3 has the log plot. The program will finish as soon as this graph is closed.

For **question 4**, the following command-line structure is expected:
```
python .\question4.py [dataset directory] [test file]
```
This question asks for the dataset directory, but we also added a second command-line argument for the *Xtest* file. This file is then converted to a pandas dataframe with the lines:
```
Xtest_file = sys.argv[2] # The second argument
Xtest = pd.read_csv(Xtest_file, header=None).values
```
So, an example call should be:
```
python .\question4.py .\datasets\ .\datasets\test.sDAT.csv
```

The *report.pdf* contains our evaluation report on the KNN models' performance. 

## Additional Files
* *utils.py*

It just helps to load the data, as this is shared by the question 1 to 3 files.