from CNF_Creator import *


def main():
    cnfC = CNF_Creator(n=50) # n is number of symbols in the 3-CNF sentence
    sentence = cnfC.CreateRandomSentence(m=120) # m is number of clauses in the 3-CNF sentence
    print('Random sentence : ',sentence)

    sentence = cnfC.ReadCNFfromCSVfile()
    print('\nSentence from CSV file : ',sentence)

#    print('\n\n')
#    print('Roll No : 2020H1030999G')
#    print('Number of clauses in CSV file : ',len(sentence))
#    print('Best model : ',[1, -2, 3, -4, -5, -6, 7, 8, 9, 10, 11, 12, -13, -14, -15, -16, -17, 18, 19, -20, 21, -22, 23, -24, 25, 26, -27, -28, 29, -30, -31, 32, 33, 34, -35, 36, -37, 38, -39, -40, 41, 42, 43, -44, -45, -46, -47, -48, -49, -50])
#    print('Fitness value of best model : 99%')
#    print('Time taken : 5.23 seconds')
#    print('\n\n')
    
if __name__=='__main__':
    main()