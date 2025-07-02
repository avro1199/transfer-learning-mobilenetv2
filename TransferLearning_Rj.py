# -*- coding: utf-8 -*-
'''

The functions and classes defined in this module will be called by a marker script. 
You should complete the functions and classes according to their specified interfaces.

No partial marks will be awarded for functions that do not meet the specifications
of the interfaces.

Last modified 2024-05-07 by Anthony Vanderkop.
Hopefully without introducing new bugs.
'''


### LIBRARY IMPORTS HERE ###
import os
import numpy as np
import keras.applications as ka  # type: ignore
import keras


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)

    '''
    # raise NotImplementedError
    return [(2020338038, 'Rj', 'Avro')]


def load_model():
    '''
    Load in a model using the tf.keras.applications model and return it.
    Insert a more detailed description here
    '''
    # raise NotImplementedError-

    model = ka.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))

    return model


def load_data(path):
    '''
    Load in the dataset from its home path. Path should be a string of the path
    to the home directory the dataset is found in. Should return a numpy array
    with paired images and class labels.

    Insert a more detailed description here.
    '''

    # raise NotImplementedError

    data = []
    labels = []
    class_names = sorted(os.listdir(path))
    class_to_idx = {class_name: idx for idx,
                    class_name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(path, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = keras.preprocessing.image.load_img(
                img_path, target_size=(224, 224))
            img = keras.preprocessing.image.img_to_array(img) / 255.0
            data.append(img)
            labels.append(class_to_idx[class_name])

    data = np.array(data)
    labels = np.array(labels)

    return (data, labels)


def split_data(X, Y, train_fraction, randomize=False, eval_set=True):
    """
    Split the data into training and testing sets. If eval_set is True, also create
    an evaluation dataset. There should be two outputs if eval_set there should
    be three outputs (train, test, eval), otherwise two outputs (train, test).

    To see what type train, test, and eval should be, refer to the inputs of 
    transfer_learning().

    Insert a more detailed description here.
    """
    # raise NotImplementedError

    if randomize:
        indices = np.random.permutation(len(X))
    else:
        indices = np.arange(len(X))

    train_size = int(len(X) * train_fraction)
    eval_test_size = len(X) - train_size

    train_indices = indices[:train_size]
    eval_test_indices = indices[train_size:]

    train_X, eval_test_X = X[train_indices], X[eval_test_indices]
    train_Y, eval_test_Y = Y[train_indices], Y[eval_test_indices]

    if eval_set:
        eval_X = eval_test_X[:(len(eval_test_X)//2)]
        eval_Y = eval_test_Y[:(len(eval_test_Y)//2)]
        test_X = eval_test_X[(len(eval_test_X)//2):]
        test_Y = eval_test_Y[(len(eval_test_Y)//2):]

        return (train_X, train_Y), (eval_X, eval_Y), (test_X, test_Y)
    else:
        return (train_X, train_Y), (eval_test_X, eval_test_Y)


def confusion_matrix(predictions, ground_truth, plot=False, all_classes=None):
    '''
    Given a set of classifier predictions and the ground truth, calculate and
    return the confusion matrix of the classifier's performance.

    Inputs:
        - predictions: np.ndarray of length n where n is the number of data
                       points in the dataset being classified and each value
                       is the class predicted by the classifier
        - ground_truth: np.ndarray of length n where each value is the correct
                        value of the class predicted by the classifier
        - plot: boolean. If true, create a plot of the confusion matrix with
                either matplotlib or with sklearn.
        - classes: a set of all unique classes that are expected in the dataset.
                   If None is provided we assume all relevant classes are in 
                   the ground_truth instead.
    Outputs:
        - cm: type np.ndarray of shape (c,c) where c is the number of unique  
              classes in the ground_truth

              Each row corresponds to a unique class in the ground truth and
              each column to a prediction of a unique class by a classifier
    '''
    # raise NotImplementedError

    if all_classes is None:
        all_classes = np.unique(ground_truth)

    cm = np.zeros((len(all_classes), len(all_classes)), dtype=int)

    class_to_index = {cls: idx for idx, cls in enumerate(all_classes)}

    for true, pred in zip(ground_truth, predictions):
        cm[class_to_index[true], class_to_index[pred]] += 1

    if plot:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=all_classes, yticklabels=all_classes)
            plt.ylabel('True Class')
            plt.xlabel('Predicted Class')
            plt.title('Confusion Matrix')
            plt.show()
        except ImportError:
            print(
                "Plotting is disabled. Install matplotlib and seaborn to enable plotting.")

    return cm


def precision(predictions, ground_truth):
    '''
    Similar to the confusion matrix, now calculate the classifier's precision

    Inputs: see confusion_matrix above
    Outputs:
        - precision: type np.ndarray of length c,
                     values are the precision for each class
    '''
    # raise NotImplementedError

    # Determine the number of classes
    classes = np.unique(ground_truth)
    num_classes = len(classes)

    # Initialize true positives (TP) and false positives (FP) arrays
    TP = np.zeros(num_classes)
    FP = np.zeros(num_classes)

    # Calculate TP and FP for each class
    for i, cls in enumerate(classes):
        TP[i] = np.sum((predictions == cls) & (ground_truth == cls))
        FP[i] = np.sum((predictions == cls) & (ground_truth != cls))

    # Calculate precision for each class
    precision = TP / (TP + FP)

    # Handle cases where TP + FP is 0 to avoid division by zero
    precision = np.nan_to_num(precision, nan=0.0)

    return precision


def recall(predictions, ground_truth):
    '''
    Similar to the confusion matrix, now calculate the classifier's recall

    Inputs: see confusion_matrix above
    Outputs:
        - recall: type np.ndarray of length c,
                     values are the recall for each class
    '''
    # raise NotImplementedError

    # Determine the number of classes
    classes = np.unique(ground_truth)
    num_classes = len(classes)

    # Initialize true positives (TP) and false negatives (FN) arrays
    TP = np.zeros(num_classes)
    FN = np.zeros(num_classes)

    # Calculate TP and FN for each class
    for i, cls in enumerate(classes):
        TP[i] = np.sum((predictions == cls) & (ground_truth == cls))
        FN[i] = np.sum((predictions != cls) & (ground_truth == cls))

    # Calculate recall for each class
    recall = TP / (TP + FN)

    # Handle cases where TP + FN is 0 to avoid division by zero
    recall = np.nan_to_num(recall, nan=0.0)

    return recall


def f1(predictions, ground_truth):
    '''
    Similar to the confusion matrix, now calculate the classifier's f1 score
    Inputs:
        - see confusion_matrix above for predictions, ground_truth
    Outputs:
        - f1: type nd.ndarry of length c where c is the number of classes
    '''

    # raise NotImplementedError

    # Calculate precision and recall
    prec = precision(predictions, ground_truth)
    rec = recall(predictions, ground_truth)

    # Calculate F1 score for each class
    f1 = 2 * (prec * rec) / (prec + rec)

    # Handle cases where precision + recall is 0 to avoid division by zero
    f1 = np.nan_to_num(f1, nan=0.0)

    return f1


def k_fold_validation(features, ground_truth, classifier, k=2):
    '''
    Inputs:
        - features: np.ndarray of features in the dataset
        - ground_truth: np.ndarray of class values associated with the features
        - fit_func: f
        - classifier: class object with both fit() and predict() methods which
        can be applied to subsets of the features and ground_truth inputs.
        - predict_func: function, calling predict_func(features) should return
        a numpy array of class predictions which can in turn be input to the 
        functions in this script to calculate performance metrics.
        - k: int, number of sub-sets to partition the data into. default is k=2
    Outputs:
        - avg_metrics: np.ndarray of shape (3, c) where c is the number of classes.
        The first row is the average precision for each class over the k
        validation steps. Second row is recall and third row is f1 score.
        - sigma_metrics: np.ndarray, each value is the standard deviation of 
        the performance metrics [precision, recall, f1_score]
    '''

    # split data
    ### YOUR CODE HERE ###
    indices = np.random.permutation(len(features))

    partition_size = int(len(features) / k)
    # eval_test_size = len(X) - train_size

    precision_scores = []
    recall_scores = []
    f1_scores = []
        
    # go through each partition and use it as a test set.
    for partition_no in range(k):
        # determine test and train sets
        ### YOUR CODE HERE###
        test_index = indices[partition_no*partition_size : (partition_no+1)*partition_size]
        # test_index = indices;
        train_index = np.array(list(index for index in indices if index not in test_index))

        print('indx => ', len(test_index))
        print('indx => ', len(train_index))

        print(test_index)
        print(train_index)

        train_features, test_features = features[train_index], features[test_index]
        train_classes, test_classes = ground_truth[train_index], ground_truth[test_index]

        # fit model to training data and perform predictions on the test set
        classifier.fit(train_features, train_classes)
        predictions = classifier.predict(test_features)
        predictions = np.array(list(prediction.argmax() for prediction in predictions))

        # calculate performance metrics
        ### YOUR CODE HERE###
        precision_values = precision(predictions, test_classes)
        recall_values = recall(predictions, test_classes)
        f1_values = f1(predictions, test_classes)

        precision_scores.append(precision_values)
        # print(precision_values)
        recall_scores.append(recall_values)
        f1_scores.append(f1_values)


    # perform statistical analyses on metrics
    ### YOUR CODE HERE###

    avg_precision = np.mean(precision_scores, axis=0)
    avg_recall = np.mean(recall_scores, axis=0)
    avg_f1 = np.mean(f1_scores, axis=0)

    sigma_precision = np.std(precision_scores, axis=0)
    sigma_recall = np.std(recall_scores, axis=0)
    sigma_f1 = np.std(f1_scores, axis=0)

    avg_metrics = np.array([avg_precision, avg_recall, avg_f1])
    sigma_metrics = np.array([sigma_precision, sigma_recall, sigma_f1])

    return avg_metrics, sigma_metrics


##################### MAIN ASSIGNMENT CODE FROM HERE ######################

# def transfer_learning(train_set, eval_set, test_set, model, parameters):
def transfer_learning():
    '''
    Implement and perform standard transfer learning here.

    Inputs:
        - train_set: list or tuple of the training images and labels in the
            form (images, labels) for training the classifier
        - eval_set: list or tuple of the images and labels used in evaluating
            the model during training, in the form (images, labels)
        - test_set: list or tuple of the training images and labels in the
            form (images, labels) for testing the classifier after training
        - model: an instance of tf.keras.applications.MobileNetV2
        - parameters: list or tuple of parameters to use during training:
            (learning_rate, momentum, nesterov)


    Outputs:
        - model : an instance of tf.keras.applications.MobileNetV2
        - metrics : list of classwise recall, precision, and f1 scores of the 
            model on the test_set (list of np.ndarray)

    '''
    input_shape = (224, 224, 3)
    num_classes = 5

    base_model = load_model()
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(
        from_logits=False), metrics=['accuracy'])
    return model
    # raise NotImplementedError
    return model, metrics


def accelerated_learning(train_set, eval_set, test_set, model, parameters):
    '''
    Implement and perform accelerated transfer learning here.

    Inputs:
        - train_set: list or tuple of the training images and labels in the
            form (images, labels) for training the classifier
        - eval_set: list or tuple of the images and labels used in evaluating
            the model during training, in the form (images, labels)
        - test_set: list or tuple of the training images and labels in the
            form (images, labels) for testing the classifier after training
        - model: an instance of tf.keras.applications.MobileNetV2
        - parameters: list or tuple of parameters to use during training:
            (learning_rate, momentum, nesterov)


    Outputs:
        - model : an instance of tf.keras.applications.MobileNetV2
        - metrics : list of classwise recall, precision, and f1 scores of the 
            model on the test_set (list of np.ndarray)

    '''
    raise NotImplementedError
    return model, metrics


if __name__ == "__main__":

    # model = load_model()
    model = transfer_learning()
    # model.summary()  # new added
    # dataset = load_data('small_flower_dataset')
    im = load_data('small_flower_dataset')

    print(im)


    model.fit(im[0], im[1], epochs=1)  # training part ###############

    # # #test-01
    output = model.predict(im[0])
    # # # print(out)
    # i = 0
    # for out in output:
    #     print(i, out.argmax(), out.max(), sep='=>')
    #     i += 1

    # test-02
    # out = model.predict(im[no][np.newaxis, ...])
    # no = 300
    # print(im[no], id[no], cl[id[no]], sep=' => ')
    # print(im[0].shape)
    # import matplotlib.pyplot as plt
    # plt.imshow(im[no])
    # plt.title(cl[id[no]])
    # plt.show()

    # img_path = 'goose.jpeg'
    # img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    # img = keras.preprocessing.image.img_to_array(img) / 255.0

    # op = model.predict(img[np.newaxis, ...])
    # print(op.argmax(), op.max(), sep='=>')
    # plt.imshow(img)
    # plt.show()

################## int code ##################
    # train_eval_test = split_data()

    # model, metrics = transfer_learning()

    # model, metrics = accelerated_learning()

    # split data test
    # x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # y = x * 100

    # o = split_data(im[0], im[1], 0.6, randomize=True, eval_set=True)
    # print(o[0])
    # print(o[1])
    # print(o[2])

    # print(type(o[1][1]))

    # print(o)
    predictions = np.array(list(out.argmax() for out in output))
    ground_truth = im[1]

    # print(predictions)
    # print(ground_truth)

    # Calculate confusion matrix without plotting
    # cm = confusion_matrix(predictions, ground_truth)
    # print("Confusion Matrix (without plot):\n", cm)

    # # print(len(cm))

    # # Calculate and plot confusion matrix
    cm_plot = confusion_matrix(predictions, ground_truth, plot=True)

    # p = precision(predictions, ground_truth)
    # r = recall(predictions, ground_truth)
    # f = f1(predictions, ground_truth)

    # print(p)
    # print(r)
    # print(f)

    # k-fold test
    # out = k_fold_validation(im[0], im[1], model, 3)
    # print('avg metrix:\n',out[0])
    # print('sigma metrix:\n',out[1])

#########################  CODE GRAVEYARD  #############################
