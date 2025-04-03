#!/usr/bin/env python3

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import dtreeviz

def main():
    input_csv = pd.read_csv('input.csv')
    # input_csv.loc[3, 'specie'] = 'maple'

    le = LabelEncoder()
    input_csv['specie_encoded'] = le.fit_transform(input_csv['specie'])

    print(input_csv)

    y = input_csv['success']
    X = input_csv.drop(['success', 'specie'], axis=1)
    clf = tree.DecisionTreeClassifier(max_depth=3)
    clf = clf.fit(X.values, y.values)

    viz_model = dtreeviz.model(clf,
        X_train=X, y_train=y,
        feature_names=X.columns.tolist(),
        target_name='success',
        class_names=list(clf.classes_),
    )

    v = viz_model.view(fontname='DejaVu Sans')
    v.save('decisiontree.svg')


if __name__=='__main__':
    main()
