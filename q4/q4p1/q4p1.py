import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
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

        self.small_image_size = 0

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
        # total no image in train/test set
        total_size = 0
        for index, class_name in enumerate(class_names):
            collection_path = base_path + '\\' + class_name
            pattern = collection_path + '\\*.jpg'

            image_paths = glob.glob(pattern)
            no_samples = len(image_paths)
            total_size += no_samples

            # read scene class image collection
            image_collection = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

            # construct scene class object to save specifications of scene class
            scene_class = BagOfVisualWords.SceneClass(index, class_name, image_collection)
            scene_classes.append(scene_class)

        return scene_classes, total_size

    def feature_descriptors(self, small_image_size=5):
        self.small_image_size = small_image_size

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

            from skimage.transform import resize

            for image in scene_class.image_collection:
                descriptors = resize(image, (self.small_image_size, self.small_image_size))
                descriptors = descriptors.reshape((self.small_image_size ** 2))
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

        try:
            vocabulary_ = retriever.retrieve()

            logging.info('vocabulary retrieved from file ... ')

        except:
            logging.info('vocabulary is going to be computed ... ')

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

        descriptors_vstack = BagOfVisualWords.stack_descriptors(descriptors)

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
        self.train_histograms = self.retrieve_histograms_of(self.train_descriptors,
                                                            Retriever(self.TRAIN_HISTOGRAMS_RETRIEVE_PATH))

        self.test_histograms = self.retrieve_histograms_of(self.test_descriptors,
                                                           Retriever(self.TEST_HISTOGRAMS_RETRIEVE_PATH))

        logging.info('histograms of train and test images computed ... ')

        return self

    def retrieve_histograms_of(self, descriptors, retriever):
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

    def classify_knn(self, k=1):

        # construct classifier model
        classifier = KNeighborsClassifier(n_neighbors=k, weights='uniform', )

        # classify image (knn)
        classifier.fit(self.train_descriptors, self.train_true_labels)

        # predict test set labels
        self.test_predicted_labels = classifier.predict(self.test_descriptors)

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
    small_image_size = 32
    k = 1

    # construct bag of visual words object
    bovw = BagOfVisualWords(train_directory_path,
                            test_directory_path,
                            KMEANS_RETRIEVE_PATH='kmeans.pkl',
                            TRAIN_HISTOGRAMS_RETRIEVE_PATH='train_histograms.pkl',
                            TEST_HISTOGRAMS_RETRIEVE_PATH='test_histograms.pkl'
                            )

    # fit model to predict labels
    test_predicted_labels = bovw. \
        initialize(). \
        feature_descriptors(small_image_size). \
        classify_knn(k=k). \
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
    plt.savefig('confusion_matrix.jpg', dpi=1200, bbox_inches='tight')

    # save configurations and result
    configs = [
        ('small_image_size: \n', small_image_size),
        ('knn k value: \n', k),
        ('accuracy: \n', accuracy)
    ]

    lines = [k + str(v) + '\n' for k, v in configs]
    f = open('configs.txt', 'w+')
    f.writelines(lines)


if __name__ == '__main__':
    main()
