import numpy as np
import random

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def inspect_confusion_matrix(examples, predictions):
    gts = [x['label'] for x in examples]
    preds = [x['label'] for x in predictions]
    # cfm = confusion_matrix(gts, preds, labels=['True', 'False', 'Neither'])
    cfm = confusion_matrix(gts, preds, labels=['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO'])
    print("Acc: {:.2f}".format(accuracy_score(gts, preds) * 100))
    print(cfm)

def filter_by_predicted_label(examples, predictions, *labels):
    if not labels:
        return examples, predictions
    exs = []
    preds = []
    for ex, p in zip(examples, predictions):
        if p["label"] not in labels:
            continue
        exs.append(ex)
        preds.append(p)
    return exs, preds

class FewshotClsReranker:
    def __init__(self):        
        # self.index_of_label = {'True': 0, 'False': 1, 'Neither': 2}
        # self.label_of_index = ['True', 'False', 'Neither']
        self.index_of_label = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2}
        self.label_of_index = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
        self.model = LogisticRegression(C=10, fit_intercept=True)        

    def train(self, examples, predictions):        
        orig_preds = np.array([self.index_of_label[p['label']] for p in predictions] )
        gt_scores = [self.index_of_label[ex['label']] for ex in examples]
        cls_scores = [p['class_probs'] for p in predictions]        
        gt_scores, cls_scores = np.array(gt_scores), np.array(cls_scores)
        
        self.model.fit(cls_scores, gt_scores)
        train_preds = self.model.predict(cls_scores)
        train_acc = np.mean(train_preds ==gt_scores)
        # print("Base ACC: {:.2f}".format(np.mean(orig_preds == gt_scores) * 100), "Train ACC: {:.2f}".format(train_acc * 100))

    def apply(self, ex, pred):
        probs = pred['class_probs']
        p = self.model.predict(np.array([probs]))[0]
        return self.label_of_index[p]


class JointClsProbExplReranker:
    def __init__(self):
        self.model = LogisticRegression(C=10, fit_intercept=True)
        self.index_of_label = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2}
        self.label_of_index = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']

    def train(self, examples, predictions):
        # calibrate true
        orig_preds = np.asarray([self.index_of_label[p['label']] for p in predictions])
        gt_scores = np.asarray([self.index_of_label[ex['label']] for ex in examples])
        cls_scores = np.asarray([p['class_probs'] for p in predictions])        
        # pre_scores = np.asarray([[p['premise_coverage']] for p in predictions])
        hyp_scores = []
        for p in predictions:
            try:
                hyp_scores.append([p['hypothesis_coverage']])
            except Exception as e:
                print(p)
                raise Exception(e)
        hyp_scores = np.asarray(hyp_scores)
        # hyp_scores = np.asarray([[p['hypothesis_coverage']] for p in predictions])
        # training_feature = np.concatenate((cls_scores, pre_scores), axis=1)
        training_feature = np.concatenate((cls_scores, hyp_scores), axis=1)

        self.model.fit(training_feature, gt_scores)
        train_preds = self.model.predict(training_feature)
        train_acc = np.mean(train_preds ==gt_scores)
        # print("Base ACC: {:.2f}".format(np.mean(orig_preds == gt_scores) * 100), "Train ACC: {:.2f}".format(train_acc * 100))

    def apply(self, ex, pred):
        # probs = pred['class_probs'] + [pred['premise_coverage']]
        probs = pred['class_probs'] + [pred['hypothesis_coverage']]
        p = self.model.predict(np.array([probs]))[0]
        return self.label_of_index[p]
