import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import glob
import scipy.spatial.distance as dist
from sklearn.metrics import accuracy_score
from pickle import load, dump, HIGHEST_PROTOCOL
import logging


class Retriever:
    class RetrieveError(Exception):
        def __init__(self, message='Object file doesnt exist to retrieve ...'):
            super().__init__(message)

    def __init__(self, path):
        self.path = path

    def save(self, to_serialize_object):
        with open(self.path, 'wb') as file:
            dump(to_serialize_object, file, protocol=HIGHEST_PROTOCOL)

    def load(self):
        with open(self.path, 'rb') as file:
            return load(file)

    def exist_file(self):
        return os.path.isfile(self.path)

    def retrieve(self):
        if self.exist_file():
            return self.load()
        else:
            raise Retriever.RetrieveError()


class BagOfVisualWords:

    def __init__(self,
                 train_directory_path,
                 test_directory_path,
                 KMEANS_RETRIEVE_PATH='kmeans.pkl',
                 TRAIN_HISTOGRAMS_RETRIEVE_PATH='train_histograms.pkl',
                 TEST_HISTOGRAMS_RETRIEVE_PATH='test_histograms.pkl'):

        self.train_directory_path = train_directory_path
        self.test_directory_path = test_directory_path

        self.dense_sift_step = 0

        self.kmeans_no_cluster = 0
        self.kmeans_n_init = 0
        self.kmeans_max_iter = 0

        self.svm_kernel = ''

        self.KMEANS_RETRIEVE_PATH = KMEANS_RETRIEVE_PATH
        self.TRAIN_HISTOGRAMS_RETRIEVE_PATH = TRAIN_HISTOGRAMS_RETRIEVE_PATH
        self.TEST_HISTOGRAMS_RETRIEVE_PATH = TEST_HISTOGRAMS_RETRIEVE_PATH

        self.train_collection_classes, self.train_set_size = None, 0
        self.test_collection_classes, self.test_set_size = None, 0
        self.classes_name = None

        self.train_descriptors, self.train_true_labels = None, None
        self.test_descriptors, self.test_true_labels = None, None

        self.vocabulary = None

        self.train_histograms = None
        self.test_histograms = None

        self.test_predicted_labels = None

    def initialize(self):

        # read scene classes from file and their specifications
        self.train_collection_classes, self.train_set_size = BagOfVisualWords.retrieve_scene_classes_of(
            self.train_directory_path)

        # read scene classes from file and their specifications
        self.test_collection_classes, self.test_set_size = BagOfVisualWords.retrieve_scene_classes_of(
            self.test_directory_path)

        # scene classes name
        self.classes_name = [train_class.class_name for train_class in self.train_collection_classes]

        logging.info('Bag of visual words object initialized ... ')

        return self

    class SceneClass:
        def __init__(self, class_label, class_name, image_collection):
            self.class_label = class_label
            self.class_name = class_name
            self.image_collection = image_collection
            self.no_samples = len(self.image_collection)

    @staticmethod
    def retrieve_scene_classes_of(path):
        directories = os.walk(path)
        base_path, class_names, _ = next(directories)
        scene_classes = []
        total_size = 0
        for index, class_name in enumerate(class_names):
            collection_path = base_path + '\\' + class_name
            pattern = collection_path + '\\*.jpg'

            image_paths = glob.glob(pattern)
            no_samples = len(image_paths)
            total_size += no_samples
            image_collection = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

            scene_class = BagOfVisualWords.SceneClass(index, class_name, image_collection)
            scene_classes.append(scene_class)

        return scene_classes, total_size

    def feature_descriptors(self, dense_sift_step=5):
        self.dense_sift_step = dense_sift_step

        # get feature descriptors of image class
        self.train_descriptors, self.train_true_labels = self.compute_feature_descriptors_of(
            self.train_collection_classes)

        # get feature descriptors of image class
        self.test_descriptors, self.test_true_labels = self.compute_feature_descriptors_of(
            self.test_collection_classes)

        logging.info('feature descriptors of train and test set extracted ... ')

        return self

    def compute_feature_descriptors_of(self, scene_classes):
        train_labels = np.array([], dtype=np.int32)
        train_descriptors = []
        algorithm: cv2.SIFT = cv2.SIFT_create()
        for scene_class in scene_classes:
            size = scene_class.no_samples
            label = scene_class.class_label
            train_labels = np.append(train_labels, np.full((size,), label))

            def dense_sift(img, step):
                key_points = [
                    cv2.KeyPoint(x, y, step)
                    for y in range(0, img.shape[0], step)
                    for x in range(0, img.shape[1], step)
                ]
                features = algorithm.compute(img, key_points)[1]
                return features

            for image in scene_class.image_collection:
                descriptors = dense_sift(image, self.dense_sift_step)
                train_descriptors.append(descriptors)

        descriptors = train_descriptors

        return descriptors, train_labels

    def build_vocabulary(self,
                         kmeans_no_clusters=400,
                         kmeans_n_init=3,
                         kmeans_max_iter=300):

        retriever = Retriever(self.KMEANS_RETRIEVE_PATH)

        self.kmeans_no_cluster = kmeans_no_clusters
        self.kmeans_n_init = kmeans_n_init
        self.kmeans_max_iter = kmeans_max_iter

        # try to retrieve vocabulary values from file
        try:
            vocabulary_ = retriever.retrieve()

            logging.info('vocabulary retrieved from file ... ')

        except:
            logging.info('vocabulary is going to be computed ... ')

            # compute vocabulary
            vocabulary_ = BagOfVisualWords.compute_vocabulary(self.train_descriptors,
                                                              kmeans_no_clusters,
                                                              kmeans_n_init,
                                                              kmeans_max_iter)
            retriever.save(vocabulary_)

            logging.info('vocabulary computed ... ')

        self.vocabulary = vocabulary_

        return self

    @staticmethod
    def compute_vocabulary(descriptors,
                           kmeans_no_clusters,
                           kmeans_n_init,
                           kmeans_max_iter):

        # stack and to numpy descriptors
        descriptors_vstack = BagOfVisualWords.stack_descriptors(descriptors)

        # clustering descriptors
        kmeans = KMeans(
            n_clusters=kmeans_no_clusters,
            max_iter=kmeans_max_iter,
            n_init=kmeans_n_init
        ).fit(descriptors_vstack)

        return kmeans.cluster_centers_

    @staticmethod
    def stack_descriptors(train_descriptors):
        train_descriptors = [en for sub_array in train_descriptors for en in sub_array]
        train_descriptors_vstack = np.vstack(train_descriptors)
        return train_descriptors_vstack

    def build_histograms(self):

        # build histogram for each image in train set
        self.train_histograms = self.retrieve_histograms_of(self.train_descriptors,
                                                            Retriever(self.TRAIN_HISTOGRAMS_RETRIEVE_PATH))

        # build histogram for each image in test set
        self.test_histograms = self.retrieve_histograms_of(self.test_descriptors,
                                                           Retriever(self.TEST_HISTOGRAMS_RETRIEVE_PATH))

        logging.info('histograms of train and test images computed ... ')

        return self

    def retrieve_histograms_of(self, descriptors, retriever):
        # try to retrieve histograms from file
        try:
            histograms_ = retriever.retrieve()

        except:

            histograms_ = self.compute_histograms(descriptors)
            retriever.save(histograms_)

        return histograms_

    def compute_histograms(self, bag_of_descriptors, metric='euclidean'):
        image_histograms = []
        vocabulary = self.vocabulary
        for class_descriptor in bag_of_descriptors:
            distance = dist.cdist(vocabulary, class_descriptor, metric)
            indices = np.argmin(distance, axis=0)

            hist, _ = np.histogram(indices, bins=self.kmeans_no_cluster)
            normal_hist = [float(i) / sum(hist) for i in hist]

            image_histograms.append(normal_hist)

        image_histograms = np.array(image_histograms)
        return image_histograms

    def classify_svm(self, kernel):
        # construct classifier model
        svc = SVC(random_state=0)

        # estimate parameters
        C_parameter = [1.0, 10.0, 100.0]
        gamma_parameter = [7.5, 10.0, 12.5, 15.0, 17.5, 20.0]

        parameters = [{'kernel': [kernel], 'C': C_parameter, 'gamma': gamma_parameter}]

        grid_search = GridSearchCV(scoring='accuracy',
                                   estimator=svc,
                                   param_grid=parameters)

        # fit model to estimate
        grid_search = grid_search.fit(self.train_histograms, self.train_true_labels)

        # fit model to train set histograms
        classifier = grid_search.best_estimator_
        classifier.fit(self.train_histograms, self.train_true_labels)

        # predict test image labels
        test_predicted_labels = classifier.predict(self.test_histograms)

        self.test_predicted_labels = test_predicted_labels

        logging.info('test set classified ')

        return self

    def get_predicted_labels(self):
        return self.test_predicted_labels

    def get_true_labels(self):
        return self.test_true_labels

    def get_classes_name(self):
        return self.classes_name


def draw_confusion_matrix(accuracy, classes_name, cm):
    plt.imshow(cm, cmap=plt.cm.Reds)
    plt.colorbar()

    plt.ylabel('Predicted Labels')
    plt.yticks(ticks=np.arange(cm.shape[0]), labels=classes_name, )

    plt.xlabel('True Labels')
    plt.xticks(ticks=np.arange(cm.shape[1]), labels=classes_name, rotation=90)

    plt.title('Confusion Matrix Normalized | Accuracy : ' + str(accuracy))


def main():
    logging.root.setLevel(logging.INFO)

    # train and test set images directory path
    train_directory_path = r'../Data/Train'
    test_directory_path = r'../Data/Test'

    # configurations
    dense_sift_step = 5
    kmeans_no_clusters = 400
    kmeans_n_init = 3
    kmeans_max_iter = 300
    kernel = 'rbf'

    # construct bag of visual words object
    bovw = BagOfVisualWords(train_directory_path,
                            test_directory_path,
                            KMEANS_RETRIEVE_PATH='kmeans.pkl',
                            TRAIN_HISTOGRAMS_RETRIEVE_PATH='train_histograms.pkl',
                            TEST_HISTOGRAMS_RETRIEVE_PATH='test_histograms.pkl')

    # fit model to predict labels
    test_predicted_labels = bovw. \
        initialize(). \
        feature_descriptors(dense_sift_step). \
        build_vocabulary(kmeans_no_clusters, kmeans_n_init, kmeans_max_iter). \
        build_histograms(). \
        classify_svm(kernel). \
        get_predicted_labels()

    # true labels
    test_true_labels = bovw.get_true_labels()

    # compute accuracy based on test true labels and test predicted labels by model
    accuracy = accuracy_score(test_true_labels, test_predicted_labels)
    print('accuracy: ', accuracy)

    # get image classes name
    classes_name = bovw.get_classes_name()

    # compute confusion matrix
    cm = confusion_matrix(test_true_labels,
                          test_predicted_labels,
                          normalize='true')

    # plot confusion matrix
    draw_confusion_matrix(accuracy, classes_name, cm)

    # save confusion matrix
    plt.savefig('res09.jpg', dpi=1200, bbox_inches='tight')

    # save configurations and result
    configs = [
        ('dense_sift_step: \n', dense_sift_step),
        ('kmeans_no_clusters: \n', kmeans_no_clusters),
        ('kmeans_n_init: \n', kmeans_n_init),
        ('kmeans_max_iter: \n', kmeans_max_iter),
        ('kernel: \n', kernel),
        ('accuracy: \n', accuracy)
    ]

    lines = [k + str(v) + '\n' for k, v in configs]
    f = open('configs.txt', 'w+')
    f.writelines(lines)


if __name__ == '__main__':
    main()
