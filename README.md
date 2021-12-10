# LSHDuplicateDetection
LSH implementation for product duplicate detection

This repostories consists of a couple of files:

main.py:
main file of the project has a main method and several functions. Functions could be split into a different file for clarity.
Contains vectorized implementation for creating a model word matrix, signature matrix, and the LSH candidates. Futhemore use sklearn for hierachical clustering

TVs-all-merged.json:
File by F. Fransicar that contains a set of products from different webshops of which some are duplicates. Is used to test the functions in main.py

results.xlsx:
results of different r,b combination

results.csv:
transformation of resutls.xlsx for plotting purposes

plot.py:
file to create plots from the data in results.csv

main.c:
first try to create C file from python using Cython. Unable to compile due to library issues
