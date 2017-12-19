from hyperbox import *
from fnc_loader import *
from utils.dataset import DataSet
from utils.score import report_score
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy as sp
import scipy.stats as stat
from iris_loader import load_iris
from random import randint
import timeit


class_label_idx = 1
sample_val_idx = 0
w2v_file = 'glove_6B_300d_w2v.txt'
w2v_dim = 300
w2v_tokens = 12
pca_features = 10
runs = 10
use_pca = False
params_tuning = True
up_sample_all = False
down_sample_all = True
recur_depth_limit = 4
box_size_gamma_params = [
    0.7,
    0.5,
    0.3,
    0.85,
]
classify_ols_activation_limit_params = [
    1000000,
]
box_size_min = 0.01
box_size_max_params = [
    0.5,
    0.3,
    0.7,
]
membership_sensitivity_params = [
    0.7,
    0.9,
    0.5,
    0.3,
]
membership_funcs = [
    min_max_membership,
    # simpsons_membership,
    # s_membership,
    # trap_membership,
]


def get_validation_mat(train_percent, train_m, train_truth):
    train_m_limit = int(train_percent * train_m.shape[0])
    new_train_m = train_m[:train_m_limit, :]
    new_train_truth = train_truth[:train_m_limit]
    new_validate_m = train_m[train_m_limit:, :]
    new_validate_truth = train_truth[train_m_limit:]
    return new_train_m, new_train_truth, new_validate_m, new_validate_truth


def is_overlapping(lower1, upper1, lower2, upper2):
    return not (lower1 > upper2 or upper1 < lower2)


def mean_confidence_interval(data, confidence=0.95):
    # code source: https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    a = 1.0*np.array(data)
    n = len(a)
    mean, std_err = np.mean(a), stat.sem(a)
    h = std_err * sp.stats.t.ppf((1+confidence)/2., n-1)
    return mean, mean-h, mean+h


def mlf_train(data, max_size, node, depth, box_size_gamma, verbose=False):
    if len(data) == 0:
        return None
    if max_size < box_size_min:
        return None
    if verbose:
        print('training at depth ' + str(depth))
        print('data size: ' + str(len(data)))
        print('max box size: ' + str(max_size))
    for sample in data:
        class_label = sample[class_label_idx]
        sample_val = sample[sample_val_idx]
        class_boxes = node.hbs[class_label]
        expand_to_cover = False
        for b in class_boxes:
            if b.can_expand(sample_val):
                b.expand(sample_val)
                expand_to_cover = True
                break
        if expand_to_cover:
            continue
        cover_box = HyperBox(len(sample_val), max_size, val=sample_val)
        class_boxes.append(cover_box)
    if verbose:
        print('class 0 hbs count: ' + str(len(node.hbs[0])))
        print('class 1 hbs count: ' + str(len(node.hbs[1])))
        print('class 2 hbs count: ' + str(len(node.hbs[2])))
        print('class 3 hbs count: ' + str(len(node.hbs[3])))
    if depth + 1 >= recur_depth_limit:
        return node
    # handle overlaps
    for b_idx_1 in range(len(node.hbs)):
        for b_idx_2 in range(len(node.hbs)):
            if b_idx_1 == b_idx_2:
                continue
            class_boxes_1 = node.hbs[b_idx_1]
            class_boxes_2 = node.hbs[b_idx_2]
            for box_1 in class_boxes_1:
                for box_2 in class_boxes_2:
                    if box_1.overlaps(box_2):
                        v, w = get_overlapping_coord(box_1, box_2)
                        ols = HyperBox(len(v), max_size)
                        ols.V = v
                        ols.W = w
                        data_recur = get_overlap_data(data, ols, b_idx_1, class_label_idx, sample_val_idx) + \
                                     get_overlap_data(data, ols, b_idx_2, class_label_idx, sample_val_idx)
                        np.random.shuffle(data_recur)
                        net = mlf_train(data_recur, max_size*box_size_gamma, MLFNode(node.num_classes), depth + 1, box_size_gamma)
                        if net is not None:
                            node.ols.append([ols, net])
    return node


def mlf_classify(memberships, node, sample, ols_limit, membership_sensitivity, func):
    for b_id in range(len(node.hbs)):
        for box in node.hbs[b_id]:
            membership = func(box, sample, membership_sensitivity)
            memberships[b_id] += membership / len(node.hbs[b_id])
    ols_i = 0
    np.random.shuffle(node.ols)
    if ols_limit < 1:
        ols_limit = ols_limit * len(node.ols)
    for b_id in range(len(node.ols)):
        ols = node.ols[b_id][0]
        if ols.contains(sample):
            if ols_i > ols_limit:
                break
            child = node.ols[b_id][1]
            mlf_classify(memberships, child, sample, ols_limit, membership_sensitivity, func)
            ols_i += 1
    rv_max = max(memberships)
    label_pred = -1
    for label_i in range(len(memberships)):
        if memberships[label_i] == rv_max:
            label_pred = label_i
            break
    return label_pred


def test_mlf(node, data, ols_limit, sensitivity, func, num_classes, verbose=True):
    start = timeit.default_timer()
    truth_labels = [t[class_label_idx] for t in data]
    truth_labels_str = [t[len(t) - 1] for t in data]
    predict_labels = []
    predict_labels_str = []
    for test_data in data:
        membership_vect = [0 for _ in range(num_classes)]
        test_val = test_data[sample_val_idx]
        label = mlf_classify(membership_vect, node, test_val, ols_limit, sensitivity, func)
        predict_labels.append(label)
        predict_labels_str.append(fnc_vect_to_label(label))
    assert len(truth_labels) == len(predict_labels)
    classify_time = timeit.default_timer() - start
    if verbose:
        print('classify time: ' + str(classify_time))
    fnc_score = report_score(truth_labels_str, predict_labels_str, verbose=verbose)
    f1_weighted = f1_score(truth_labels, predict_labels, average='weighted')
    f1_class = f1_score(truth_labels, predict_labels, average=None)
    acc = accuracy_score(truth_labels, predict_labels)
    print('MLF f1 comp: ' + str(f1_weighted) + ', ' + str(f1_class))
    print('MLF acc: ' + str(acc))
    return fnc_score, predict_labels, f1_weighted, acc


def run_rand_classifier(train_m, train_truth, test_m, test_truth):
    pred_strs = [fnc_vect_to_label(randint(0, 3)) for _ in test_truth]
    test_strs = [fnc_vect_to_label(d) for d in test_truth]
    print('Rand results:')
    return report_score(test_strs, pred_strs)


def run_base_classifier(train_m, train_truth, test_m, test_truth):
    print('Base results:')
    pred_strs = ['unrelated' for _ in test_truth]
    test_strs = [fnc_vect_to_label(d) for d in test_truth]
    s = report_score(test_strs, pred_strs)
    pred_strs = ['agree' for _ in test_truth]
    test_strs = [fnc_vect_to_label(d) for d in test_truth]
    s = max(s, report_score(test_strs, pred_strs))
    return s


def run_NB_classifier(train_m, train_truth, test_m, test_truth, validate=False):
    print('running NB classifier, validate=' + str(validate))
    priors = [None, [0.5, 0.5]]
    best_prior = None
    highest_fnc = 0
    if validate:
        split_train_m, split_train_truth, validate_m, validate_truth = get_validation_mat(0.7, train_m, train_truth)
        # split_train_truth = label_binarize(split_train_truth, classes=[0, 1, 2, 3])
        for pr in priors:
            model = GaussianNB(priors=pr)
            clf = OneVsOneClassifier(model)
            pred_labels = clf.fit(split_train_m, split_train_truth).predict(validate_m).tolist()
            # pred_strs = fnc_one_hot_label_decode(pred_labels)
            pred_strs = [fnc_vect_to_label(d) for d in pred_labels]
            test_strs = [fnc_vect_to_label(d) for d in validate_truth]
            fnc_s = report_score(test_strs, pred_strs, verbose=False)
            if fnc_s > highest_fnc:
                highest_fnc = fnc_s
                best_prior = pr
    # train_truth = label_binarize(train_truth, classes=[0, 1, 2, 3])
    model = GaussianNB(priors=best_prior)
    clf = OneVsOneClassifier(model)
    pred_labels = clf.fit(train_m, train_truth).predict(test_m).tolist()
    pred_strs = [fnc_vect_to_label(d) for d in pred_labels]
    test_strs = [fnc_vect_to_label(d) for d in test_truth]
    print('NB results:')
    f1_w = f1_score(test_strs, pred_strs, average='weighted')
    f1_class = f1_score(test_strs, pred_strs, average=None)
    acc = accuracy_score(test_strs, pred_strs)
    print('NB f1 comp: ' + str(f1_w) + ', ' + str(f1_class))
    print('NB acc: ' + str(acc))
    return report_score(test_strs, pred_strs), f1_w, acc


def run_LR_classifier(train_m, train_truth, test_m, test_truth, validate=False):
    print('running LR classifier, validate=' + str(validate))
    cs = [0.01, 0.1, 1.0, 10]
    regs = ['l2']
    highest_fnc = 0
    best_c = 0.1
    best_reg = 'l2'
    if validate:
        split_train_m, split_train_truth, validate_m, validate_truth = get_validation_mat(0.7, train_m, train_truth)
        for c in cs:
            for reg in regs:
                model = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=c, penalty=reg)
                model.fit(split_train_m, split_train_truth)
                pred_labels = model.predict(validate_m)
                pred_strs = [fnc_vect_to_label(d) for d in pred_labels]
                test_strs = [fnc_vect_to_label(d) for d in validate_truth]
                fnc_s = report_score(test_strs, pred_strs, verbose=False)
                if fnc_s > highest_fnc:
                    highest_fnc = fnc_s
                    best_reg = reg
                    best_c = c
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=best_c, penalty=best_reg)
    model.fit(train_m, train_truth)
    pred_labels = model.predict(test_m)
    pred_strs = [fnc_vect_to_label(d) for d in pred_labels]
    test_strs = [fnc_vect_to_label(d) for d in test_truth]
    print('LR results:')
    f1_w = f1_score(test_strs, pred_strs, average='weighted')
    f1_class = f1_score(test_strs, pred_strs, average=None)
    acc = accuracy_score(test_strs, pred_strs)
    print('LR f1 comp: ' + str(f1_w) + ', ' + str(f1_class))
    print('LR acc: ' + str(acc))
    return report_score(test_strs, pred_strs), f1_w, acc


def run_NN_classifier(train_m, train_truth, test_m, test_truth, validate=False):
    print('running NN classifier, validate=' + str(validate))
    hidden_layer_sizes = [5, 10, 50, 100, 1000]
    activations = ['identity', 'logistic', 'tanh', 'relu']
    learning_rates = [0.001, 0.01, 0.0001]
    highest_fnc = 0
    best_h_size = 100
    best_activation = 'relu'
    best_lr = 0.001
    if validate:
        split_train_m, split_train_truth, validate_m, validate_truth = get_validation_mat(0.7, train_m, train_truth)
        # split_train_truth = label_binarize(split_train_truth, classes=[0, 1, 2, 3])
        for h in hidden_layer_sizes:
            for a in activations:
                for lr in learning_rates:
                    model = MLPClassifier(max_iter=1000, hidden_layer_sizes=h, activation=a, learning_rate_init=lr)
                    clf = OneVsOneClassifier(model)
                    pred_labels = clf.fit(split_train_m, split_train_truth).predict(validate_m).tolist()
                    # pred_strs = fnc_one_hot_label_decode(pred_labels)
                    pred_strs = [fnc_vect_to_label(d) for d in pred_labels]
                    test_strs = [fnc_vect_to_label(d) for d in validate_truth]
                    fnc_s = report_score(test_strs, pred_strs, verbose=False)
                    if fnc_s > highest_fnc:
                        highest_fnc = fnc_s
                        best_activation = a
                        best_h_size = h
                        best_lr = lr
    # train_truth = label_binarize(train_truth, classes=[0, 1, 2, 3])
    model = MLPClassifier(max_iter=1000, hidden_layer_sizes=best_h_size, activation=best_activation, learning_rate_init=best_lr)
    clf = OneVsOneClassifier(model)
    pred_labels = clf.fit(train_m, train_truth).predict(test_m).tolist()
    pred_strs = [fnc_vect_to_label(d) for d in pred_labels]
    test_strs = [fnc_vect_to_label(d) for d in test_truth]
    print('NN results:')
    f1_w = f1_score(test_strs, pred_strs, average='weighted')
    f1_class = f1_score(test_strs, pred_strs, average=None)
    acc = accuracy_score(test_strs, pred_strs)
    print('NN f1 comp: ' + str(f1_w) + ', ' + str(f1_class))
    print('NN acc: ' + str(acc))
    return report_score(test_strs, pred_strs), f1_w, acc


def run_DT_classifier(train_m, train_truth, test_m, test_truth, validate=False):
    print('running DT classifier, validate='+str(validate))
    criterions = ['gini', 'entropy']
    splitters = ['best', 'random']
    max_features = [
        None,
        0.5,
        0.9,
        'sqrt', 'log2'
    ]
    max_depths = [
        None,
        5, 10, 50, 100
    ]
    highest_fnc = 0
    best_depth = None
    best_criterion = 'gini'
    best_features = None
    best_splitter = 'best'
    if validate:
        split_train_m, split_train_truth, validate_m, validate_truth = get_validation_mat(0.7, train_m, train_truth)
        # split_train_truth = label_binarize(split_train_truth, classes=[0, 1, 2, 3])
        for c in criterions:
            for s in splitters:
                for f in max_features:
                    for d in max_depths:
                        model = DecisionTreeClassifier(criterion=c, splitter=s, max_features=f, max_depth=d)
                        clf = OneVsOneClassifier(model)
                        pred_labels = clf.fit(split_train_m, split_train_truth).predict(validate_m).tolist()
                        # pred_strs = fnc_one_hot_label_decode(pred_labels)
                        pred_strs = [fnc_vect_to_label(d) for d in pred_labels]
                        test_strs = [fnc_vect_to_label(d) for d in validate_truth]
                        fnc_s = report_score(test_strs, pred_strs, verbose=False)
                        if fnc_s > highest_fnc:
                            highest_fnc = fnc_s
                            best_criterion = c
                            best_splitter = s
                            best_features = f
                            best_depth = d
    # train_truth = label_binarize(train_truth, classes=[0, 1, 2, 3])
    model = DecisionTreeClassifier(criterion=best_criterion, splitter=best_splitter, max_features=best_features, max_depth=best_depth)
    clf = OneVsOneClassifier(model)
    pred_labels = clf.fit(train_m, train_truth).predict(test_m).tolist()
    pred_strs = [fnc_vect_to_label(d) for d in pred_labels]
    test_strs = [fnc_vect_to_label(d) for d in test_truth]
    print('DT results:')
    f1_w = f1_score(test_strs, pred_strs, average='weighted')
    f1_class = f1_score(test_strs, pred_strs, average=None)
    acc = accuracy_score(test_strs, pred_strs)
    print('DT f1 comp: ' + str(f1_w) + ', ' + str(f1_class))
    print('DT acc: ' + str(acc))
    return report_score(test_strs, pred_strs), f1_w, acc


def run_two_pass_MLF_classifier(train_list, train_matrix, test_list, test_matrix, label_idx, validate=False):
    binary_train_list = []
    stance_train_list = []
    binary_test_list = []
    for val in train_list:
        if val[label_idx] != 0:
            stance_train_list.append(val)
            binary_train_list.append([val[0], 1, val[2], val[3], val[4]])
        else:
            binary_train_list.append([val[0], 0, val[2], val[3], val[4]])
    for val in test_list:
        if val[label_idx] != 0:
            binary_test_list.append([val[0], 1, val[2], val[3], val[4]])
        else:
            binary_test_list.append([val[0], 0, val[2], val[3], val[4]])
    stance_train_mat = np.zeros((len(stance_train_list), train_matrix.shape[1]))
    for j in range(len(stance_train_list)):
        stance_train_mat[j, :] = np.array(stance_train_list[j][0])
    clf_relation, preds, rs, f1, acc = run_single_MLF_classifier(binary_train_list, train_matrix, binary_test_list,
                                                    test_matrix, label_idx, num_classes=2, validate=validate)
    stance_test_list = []
    for j in range(len(preds)):
        if preds[j] != 0:
            stance_test_list.append(test_list[j])
    stance_test_mat = np.zeros((len(stance_test_list), test_matrix.shape[1]))
    for j in range(len(stance_test_list)):
        stance_test_mat[j, :] = np.array(stance_test_list[j][0])
    clf_stance, preds, rs, f1, acc = run_single_MLF_classifier(stance_train_list, stance_train_mat, stance_test_list, stance_test_mat,
                                                    label_idx, num_classes=4, validate=validate)
    return clf_relation, clf_stance, preds, rs


def run_single_MLF_classifier(train_list, train_matrix, test_list, test_matrix, label_idx, num_classes, validate=False):
    curr_highest_valid_score = 0
    best_b_size_gamma = 0.7
    best_ols_lim = 50000
    best_b_max = 0.5
    best_meb_sens = 0.7
    best_func_idx = 0
    max_train_val = np.amax(train_matrix)
    min_train_val = np.amin(train_matrix)
    train_val_range = abs(max_train_val - min_train_val)
    max_test_val = np.amax(test_matrix)
    min_test_val = np.amin(test_matrix)
    test_val_range = abs(max_test_val - min_test_val)
    print('begin mlf training, train mat value range: ' + str(train_val_range)
          + ' test mat value range: ' + str(test_val_range))
    train_start_time = timeit.default_timer()
    if validate:
        train_subset, valid_subset = split_by_ratio(0.7, train_list, label_idx)
        for b_size_gamma in box_size_gamma_params:
            print('tuning b_size_gamma ' + str(best_b_size_gamma))
            for ols_lim in classify_ols_activation_limit_params:
                print('tuning ols_lim ' + str(ols_lim))
                for b_max in box_size_max_params:
                    print('tuning b_max ' + str(b_max))
                    for meb_sens in membership_sensitivity_params:
                        print('tuning meb_sens ' + str(meb_sens))
                        for func_idx in range(len(membership_funcs)):
                            print('tuning func ' + str(func_idx))
                            r = mlf_train(train_subset, b_max * train_val_range, MLFNode(num_classes), 0, b_size_gamma)
                            score, _, _, _ = test_mlf(r, valid_subset, ols_lim, meb_sens, membership_funcs[func_idx], num_classes,
                                                    verbose = False)
                            if score > curr_highest_valid_score:
                                curr_highest_valid_score = score
                                best_b_size_gamma = b_size_gamma
                                best_ols_lim = ols_lim
                                best_b_max = b_max
                                best_meb_sens = meb_sens
                                best_func_idx = func_idx
                                print('new highest score: ' + str(score))
                                print('best box size gamma: ' + str(best_b_size_gamma))
                                print('best ols lim: ' + str(best_ols_lim))
                                print('best box max size: ' + str(best_b_max))
                                print('best sensitivity: ' + str(best_meb_sens))
                                print('best func: ' + str(func_idx))
    r = mlf_train(train_list, best_b_max * train_val_range, MLFNode(num_classes), 0, best_b_size_gamma)
    train_time = timeit.default_timer() - train_start_time
    print('training time: ' + str(train_time))
    print('testing MLF')
    rs, preds, f1, acc = test_mlf(r, test_list, best_ols_lim, best_meb_sens, membership_funcs[best_func_idx],
                                  num_classes, verbose=True)
    return r, preds, rs, f1, acc

def run_iris_test():
    train_set, test_set = load_iris('iris.txt', 0.5)
    truth_labels = [t[class_label_idx] for t in test_set]
    predict_labels = []
    start_time_iris = timeit.default_timer()
    rooti = mlf_train(train_set, 3.0, MLFNode(3), 0, 0.85)
    for test_data in test_set:
        membership_vect = [0 for _ in range(3)]
        test_val = test_data[sample_val_idx]
        label = mlf_classify(membership_vect, rooti, test_val, 5000, 0.7, membership_funcs[0])
        predict_labels.append(label)
    assert len(truth_labels) == len(predict_labels)
    iris_time = timeit.default_timer() - start_time_iris
    print('iris time: ' + str(iris_time))
    f1_w = f1_score(truth_labels, predict_labels, average='weighted')
    f1_class = f1_score(truth_labels, predict_labels, average=None)
    acc = accuracy_score(truth_labels, predict_labels)
    print('iris f1: ' + str(f1_w) + ', ' + str(f1_class))
    print('iris acc: ' + str(acc))


class MLFNode:

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.ols = []
        self.hbs = [
            [] for _ in range(num_classes)
        ]

    def print_info(self):
        print('MLFNode info:')
        print('number of classes: ' + str(self.num_classes))
        for idx in range(self.num_classes):
            print('HBS class ' + str(idx) + ' size: ' + str(len(self.hbs[idx])))
            if len(self.hbs[idx]) > 0:
                print('sample HBS for class ' + str(idx) + ':')
                self.hbs[idx][0].print_info()
        print('OLS size: ' + str(len(self.ols)))
        if len(self.ols) > 0:
            print(' ')
            print('sample child node: ')
            self.ols[0][1].print_info()

def write_data_to_file(datalist, filename):
    f = open(filename, 'w', encoding='utf-8')
    for d in datalist:
        f.write(str(d))
        f.write('\n')
    f.close()


if __name__ == '__main__':
    run_iris_test()
    start_time = timeit.default_timer()
    train_dataset = DataSet()
    # Load the competition dataset
    competition_dataset = DataSet("competition_test")
    load_time = timeit.default_timer() - start_time
    print('data load time: ' + str(load_time))

    # start_time = timeit.default_timer()
    # tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    # train_vect, sample_val_idx, class_label_idx, train_mat = generate_tfidf_vect(train_dataset, tfidf)
    # test_vect, sample_val_idx, class_label_idx, test_mat = generate_tfidf_vect(competition_dataset, tfidf, training=False)
    # tfidf_time = timeit.default_timer() - start_time
    # print('tfidf time: ' + str(tfidf_time))

    lmtzr = WordNetLemmatizer()
    glove = gensim.models.KeyedVectors.load_word2vec_format(w2v_file, binary=False)

    results_lr = []
    results_nb = []
    results_nn = []
    results_dt = []
    results_smlf = []
    f1s_lr = []
    f1s_nb = []
    f1s_nn = []
    f1s_dt = []
    f1s_smlf = []
    acc_lr = []
    acc_nb = []
    acc_nn = []
    acc_dt = []
    acc_smlf = []
    time_lr = []
    time_nb = []
    time_nn = []
    time_dt = []
    time_smlf = []
    for run in range(runs):
        print('\nrun ' + str(run))
        start_time = timeit.default_timer()
        print('training data glove')
        train_vect, sample_val_idx, class_label_idx, train_mat = generate_fnc_features(w2v_file,
                                                                                    train_dataset.stances,
                                                                                    train_dataset.articles,
                                                                                    dim=w2v_dim, max_token=w2v_tokens,
                                                                                    glove=glove, lmtzr=lmtzr, strategy='paircosine',
                                                                                    up_sample=up_sample_all, down_sample=down_sample_all)
        print('testing data glove')
        test_vect, sample_val_idx, class_label_idx, test_mat = generate_fnc_features(w2v_file,
                                                                                  competition_dataset.stances,
                                                                                  competition_dataset.articles,
                                                                                  dim=w2v_dim, max_token=w2v_tokens,
                                                                                  glove=glove, lmtzr=lmtzr, strategy='paircosine',
                                                                                  up_sample=up_sample_all, down_sample=down_sample_all)
        glove_time = timeit.default_timer() - start_time
        print('glove time: ' + str(glove_time))

        if use_pca:
            start_time = timeit.default_timer()
            train_pca_mat = PCA(n_components=pca_features).fit_transform(train_mat)
            test_pca_mat = PCA(n_components=pca_features).fit_transform(test_mat)
            for i in range(len(train_vect)):
                train_vect[i][sample_val_idx] = train_pca_mat[i, :].tolist()
            for i in range(len(test_vect)):
                test_vect[i][sample_val_idx] = test_pca_mat[i, :].tolist()
            train_mat = train_pca_mat
            test_mat = test_pca_mat
            pca_time = timeit.default_timer() - start_time
            print('PCA time: ' + str(pca_time))

        train_truth_labels = [d[class_label_idx] for d in train_vect]
        test_truth_labels = [d[class_label_idx] for d in test_vect]

        run_rand_classifier(train_mat, train_truth_labels, test_mat, test_truth_labels)
        run_base_classifier(train_mat, train_truth_labels, test_mat, test_truth_labels)
        start_time = timeit.default_timer()
        lr_s, lr_f1, lr_acc = run_LR_classifier(train_mat, train_truth_labels, test_mat, test_truth_labels, validate=params_tuning)
        lr_time = timeit.default_timer() - start_time
        time_lr.append(lr_time)
        start_time = timeit.default_timer()
        nb_s, nb_f1, nb_acc  = run_NB_classifier(train_mat, train_truth_labels, test_mat, test_truth_labels, validate=params_tuning)
        nb_time = timeit.default_timer() - start_time
        time_nb.append(nb_time)
        start_time = timeit.default_timer()
        nn_s, nn_f1, nn_acc  = run_NN_classifier(train_mat, train_truth_labels, test_mat, test_truth_labels, validate=params_tuning)
        nn_time = timeit.default_timer() - start_time
        time_nn.append(nn_time)
        start_time = timeit.default_timer()
        dt_s, dt_f1, dt_acc  = run_DT_classifier(train_mat, train_truth_labels, test_mat, test_truth_labels, validate=params_tuning)
        dt_time = timeit.default_timer() - start_time
        time_dt.append(dt_time)
        print('running single MLF')
        start_time = timeit.default_timer()
        root, p, smlf_s, smlf_f1, smlf_acc = run_single_MLF_classifier(train_vect, train_mat, test_vect, test_mat,
                                                                       class_label_idx, 4, validate=params_tuning)
        smlf_time = timeit.default_timer() - start_time
        time_smlf.append(smlf_time)
        root.print_info()
        results_lr.append(lr_s)
        results_nb.append(nb_s)
        results_nn.append(nn_s)
        results_dt.append(dt_s)
        results_smlf.append(smlf_s)
        f1s_lr.append(lr_f1)
        f1s_nb.append(nb_f1)
        f1s_nn.append(nn_f1)
        f1s_dt.append(dt_f1)
        f1s_smlf.append(smlf_f1)
        acc_lr.append(lr_acc)
        acc_nb.append(nb_acc)
        acc_nn.append(nn_acc)
        acc_dt.append(dt_acc)
        acc_smlf.append(smlf_acc)
    print('lr: ' + str(np.mean(results_lr)) + ' +- ' +str(np.std(results_lr)) + ' time: ' + str(np.mean(time_lr)))
    print('nb: ' + str(np.mean(results_nb)) + ' +- ' + str(np.std(results_nb)) + ' time: ' + str(np.mean(time_nb)))
    print('nn: ' + str(np.mean(results_nn)) + ' +- ' + str(np.std(results_nn)) + ' time: ' + str(np.mean(time_nn)))
    print('dt: ' + str(np.mean(results_dt)) + ' +- ' + str(np.std(results_dt)) + ' time: ' + str(np.mean(time_dt)))
    print('smlf: ' + str(np.mean(results_smlf)) + ' +- ' + str(np.std(results_smlf)) + ' time: ' + str(np.mean(time_smlf)))
    cache_dir = 'cache_func_0'
    write_data_to_file(results_lr, cache_dir + '/lr_fnc.txt')
    write_data_to_file(results_nb, cache_dir + '/nb_fnc.txt')
    write_data_to_file(results_nn, cache_dir + '/nn_fnc.txt')
    write_data_to_file(results_dt, cache_dir + '/dt_fnc.txt')
    write_data_to_file(results_smlf, cache_dir + '/smlf_fnc.txt')
    write_data_to_file(f1s_lr, cache_dir + '/lr_f1.txt')
    write_data_to_file(f1s_nb, cache_dir + '/nb_f1.txt')
    write_data_to_file(f1s_nn, cache_dir + '/nn_f1.txt')
    write_data_to_file(f1s_dt, cache_dir + '/dt_f1.txt')
    write_data_to_file(f1s_smlf, cache_dir + '/smlf_f1.txt')
    write_data_to_file(acc_lr, cache_dir + '/lr_acc.txt')
    write_data_to_file(acc_nb, cache_dir + '/nb_acc.txt')
    write_data_to_file(acc_nn, cache_dir + '/nn_acc.txt')
    write_data_to_file(acc_dt, cache_dir + '/dt_acc.txt')
    write_data_to_file(acc_smlf, cache_dir + '/smlf_acc.txt')
    write_data_to_file(time_lr, cache_dir + '/lr_time.txt')
    write_data_to_file(time_nb, cache_dir + '/nb_time.txt')
    write_data_to_file(time_nn, cache_dir + '/nn_time.txt')
    write_data_to_file(time_dt, cache_dir + '/dt_time.txt')
    write_data_to_file(time_smlf, cache_dir + '/smlf_time.txt')

    lr_m_f1, lr_lower_f1, lr_upper_f1 = mean_confidence_interval(f1s_lr, 0.95)
    nb_m_f1, nb_lower_f1, nb_upper_f1 = mean_confidence_interval(f1s_nb, 0.95)
    dt_m_f1, dt_lower_f1, dt_upper_f1 = mean_confidence_interval(f1s_dt, 0.95)
    nn_m_f1, nn_lower_f1, nn_upper_f1 = mean_confidence_interval(f1s_nn, 0.95)
    smlf_m_f1, smlf_lower_f1, smlf_upper_f1 = mean_confidence_interval(f1s_smlf, 0.95)
    lr_m_acc, lr_lower_acc, lr_upper_acc = mean_confidence_interval(acc_lr, 0.95)
    nb_m_acc, nb_lower_acc, nb_upper_acc = mean_confidence_interval(acc_nb, 0.95)
    dt_m_acc, dt_lower_acc, dt_upper_acc = mean_confidence_interval(acc_dt, 0.95)
    nn_m_acc, nn_lower_acc, nn_upper_acc = mean_confidence_interval(acc_nn, 0.95)
    smlf_m_acc, smlf_lower_acc,smlf_upper_acc = mean_confidence_interval(acc_smlf, 0.95)
    lr_m_fnc, lr_lower_fnc, lr_upper_fnc = mean_confidence_interval(results_lr, 0.95)
    nb_m_fnc, nb_lower_fnc, nb_upper_fnc = mean_confidence_interval(results_nb, 0.95)
    dt_m_fnc, dt_lower_fnc, dt_upper_fnc = mean_confidence_interval(results_dt, 0.95)
    nn_m_fnc, nn_lower_fnc, nn_upper_fnc = mean_confidence_interval(results_nn, 0.95)
    smlf_m_fnc, smlf_lower_fnc,smlf_upper_fnc = mean_confidence_interval(results_smlf, 0.95)

    print('mean f1 for logistic regression: ' + str(lr_m_f1))
    print('mean f1 for naive bayes: ' + str(nb_m_f1))
    print('mean f1 for decision tree: ' + str(dt_m_f1))
    print('mean f1 for nn: ' + str(nn_m_f1))
    print('mean f1 for mlf: ' + str(smlf_m_f1))

    print('mean acc for logistic regression: ' + str(lr_m_acc))
    print('mean acc for naive bayes: ' + str(nb_m_acc))
    print('mean acc for decision tree: ' + str(dt_m_acc))
    print('mean acc for nn: ' + str(nn_m_acc))
    print('mean acc for mlf: ' + str(smlf_m_acc))

    print('f1 confidence interval for logistic regression: ' + str(lr_lower_f1) + ' ' + str(lr_upper_f1))
    print('f1 confidence interval for naive bayes: ' + str(nb_lower_f1) + ' ' + str(nb_upper_f1))
    print('f1 confidence interval for decision tree: ' + str(dt_lower_f1) + ' ' + str(dt_upper_f1))
    print('f1 confidence interval for nn: ' + str(nn_lower_f1) + ' ' + str(nn_upper_f1))
    print('f1 confidence interval for mlf: ' + str(smlf_lower_f1) + ' ' + str(smlf_upper_f1))

    print('acc confidence interval for logistic regression: ' + str(lr_lower_acc) + ' ' + str(lr_upper_acc))
    print('acc confidence interval for naive bayes: ' + str(nb_lower_acc) + ' ' + str(nb_upper_acc))
    print('acc confidence interval for decision tree: ' + str(dt_lower_acc) + ' ' + str(dt_upper_acc))
    print('acc confidence interval for nn: ' + str(nn_lower_acc) + ' ' + str(nn_upper_acc))
    print('acc confidence interval for mlf: ' + str(smlf_lower_acc) + ' ' + str(smlf_upper_acc))

    print('fnc confidence interval for logistic regression: ' + str(lr_lower_fnc) + ' ' + str(lr_upper_fnc))
    print('fnc confidence interval for naive bayes: ' + str(nb_lower_fnc) + ' ' + str(nb_upper_fnc))
    print('fnc confidence interval for decision tree: ' + str(dt_lower_fnc) + ' ' + str(dt_upper_fnc))
    print('fnc confidence interval for nn: ' + str(nn_lower_fnc) + ' ' + str(nn_upper_fnc))
    print('fnc confidence interval for mlf: ' + str(smlf_lower_fnc) + ' ' + str(smlf_upper_fnc))

    if is_overlapping(dt_lower_f1, dt_upper_f1, smlf_lower_f1, smlf_upper_f1):
        print('decision tree and mlf f1 confidence intervals overlap')
    else:
        print('decision tree and mlf f1 confidence intervals do not overlap')
    if is_overlapping(lr_lower_f1, lr_upper_f1, smlf_lower_f1, smlf_upper_f1):
        print('logistic regression and mlf f1 confidence intervals overlap')
    else:
        print('logistic regression and mlf f1 confidence intervals do not overlap')
    if is_overlapping(nb_lower_f1, nb_upper_f1, smlf_lower_f1, smlf_upper_f1):
        print('naive bayes and mlf f1 confidence intervals overlap')
    else:
        print('naive bayes and mlf f1 confidence intervals do not overlap')
    if is_overlapping(smlf_lower_f1, smlf_upper_f1, nn_lower_f1, nn_upper_f1):
        print('mlf and nn f1 confidence intervals overlap')
    else:
        print('mlf and nn f1 confidence intervals do not overlap')

    if is_overlapping(dt_lower_acc, dt_upper_acc, smlf_lower_acc, smlf_upper_acc):
        print('decision tree and mlf acc confidence intervals overlap')
    else:
        print('decision tree and mlf acc confidence intervals do not overlap')
    if is_overlapping(lr_lower_acc, lr_upper_acc, smlf_lower_acc, smlf_upper_acc):
        print('logistic regression and mlf acc confidence intervals overlap')
    else:
        print('logistic regression and mlf acc confidence intervals do not overlap')
    if is_overlapping(nb_lower_acc, nb_upper_acc, smlf_lower_acc, smlf_upper_acc):
        print('naive bayes and mlf acc confidence intervals overlap')
    else:
        print('naive bayes and mlf acc confidence intervals do not overlap')
    if is_overlapping(smlf_lower_acc, smlf_upper_acc, nn_lower_acc, nn_upper_acc):
        print('mlf and nn acc confidence intervals overlap')
    else:
        print('mlf and nn acc confidence intervals do not overlap')

    if is_overlapping(dt_lower_fnc, dt_upper_fnc, smlf_lower_fnc, smlf_upper_fnc):
        print('decision tree and mlf fnc confidence intervals overlap')
    else:
        print('decision tree and mlf fnc confidence intervals do not overlap')
    if is_overlapping(lr_lower_fnc, lr_upper_fnc, smlf_lower_fnc, smlf_upper_fnc):
        print('logistic regression and mlf fnc confidence intervals overlap')
    else:
        print('logistic regression and mlf fnc confidence intervals do not overlap')
    if is_overlapping(nb_lower_fnc, nb_upper_fnc, smlf_lower_fnc, smlf_upper_fnc):
        print('naive bayes and mlf acc confidence intervals overlap')
    else:
        print('naive bayes and mlf acc confidence intervals do not overlap')
    if is_overlapping(smlf_lower_fnc, smlf_upper_fnc, nn_lower_fnc, nn_upper_fnc):
        print('mlf and nn fnc confidence intervals overlap')
    else:
        print('mlf and nn fnc confidence intervals do not overlap')

    # anova tests
    f_val_f1, p_val_f1 = stat.f_oneway(f1s_lr, f1s_nb, f1s_nn, f1s_dt, f1s_smlf)
    print('f1 p val = ' + str(p_val_f1))

    f_val_acc, p_val_acc = stat.f_oneway(acc_lr, acc_nb, acc_nn, acc_dt, acc_smlf)
    print('acc p val = ' + str(p_val_acc))

    f_val_fnc, p_val_fnc = stat.f_oneway(results_lr, results_nb, results_nn, results_dt, results_smlf)
    print('fnc p val = ' + str(p_val_fnc))
