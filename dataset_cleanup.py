import numpy as np

#This function takes an x and y and returns an even amount of every category.
#Discards everything else. RETURNS VALUES IN ORDER, UNDOES ALL RANDOMIZATION!!!!
#if max is 0 then it returns as much as possible, otherwise it attempts to return either max of each category or as much as possible, whichever is smaller
def even_selection(X, y, max=0):
    # get maximum possible size
    y = np.array(y)
    max_possible = -1
    for i in range(5):
        count = np.count_nonzero(y == i+1)
        if max_possible > count or max_possible == -1:
            max_possible = count
    
    #take max possible or max whichevers smaller
    if max != 0 and max_possible > max:
        max_possible = max

    #get indices of occurences
    indices = []
    new_X = []
    new_y = []
    for i in range(5):
        mew = np.where(y == i+1)[0]
        mew = mew[:max_possible]
        indices.append(mew)

    indices = np.array(indices).flatten()

    #get occurences
    for i in indices:
        new_X.append(X[i])
        new_y.append(y[i])

    return new_X, new_y