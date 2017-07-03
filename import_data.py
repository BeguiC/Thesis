import numpy as np
from scipy import misc



def get_classification_data(filename, total_size):
    n_images = 40
    X = np.zeros((total_size,49, 49))
    Y = []

    parts = ["LargeBalanced", "MiddleBalanced", "SmallBalanced", "TransversalBalanced"]

    def import_part(part_index):
        for i in range(n_images):x
            img = misc.imread(filename + "/"+ parts[part_index]+ "/" + str(i) + ".jpg",flatten=True)
            X[part_index * n_images + i] = img
            Y.append(part_index)
        
        print (parts[part_index] + ' loaded')

    import_part(0)
    import_part(1)
    import_part(2)
    import_part(3)


    X = X.astype(np.float32) / 255
    Y = np.array(Y).astype(np.int32)

    return X, Y


def get_randomized_classification_data(filename, xp, training_size, test_size):
    # -- Importation of the data and division between training set and test set --#
    total_size = 160
    x_data, y_data = get_classification_data(filename, total_size)

    Xtr = []
    Xte = []
    Ytr = []
    Yte = []

    indexes = np.random.permutation(total_size)

    for i in range(total_size):
        if i < 120:
            Xtr.append(x_data[indexes[i]])
            Ytr.append(y_data[indexes[i]])
        else:
            Xte.append(x_data[indexes[i]])
            Yte.append(y_data[indexes[i]])
    Xtr = xp.asarray(Xtr)
    Xte = xp.asarray(Xte)
    Ytr = xp.asarray(Ytr)
    Yte = xp.asarray(Yte)
    return [Xtr, Ytr, Xte, Yte]
