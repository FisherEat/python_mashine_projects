# It is a mashine learning test by gaolong

'''

本案例是tensorflow预测鸢尾花模型的案例的实现部分

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import mashine_data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int, help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])
    (train_x, train_y), (test_x, test_y) = mashine_data.load_data()
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
        print(tf.feature_column.numeric_column(key=key))

    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[10, 10],
        n_classes=3)
    classifier.train(
        input_fn=lambda:mashine_data.train_input_fn(train_x, train_y, args.batch_size),
        steps=args.train_steps)
    print(argv)
    eval_result = classifier.evaluate(
        input_fn=lambda:mashine_data.eval_input_fn(test_x, test_y,
                                                args.batch_size))
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 5.9],
        'SepalWidth': [3.3, 3.0, 3.0],
        'PetalLength': [1.7, 4.2, 4.2],
        'PetalWidth': [0.5, 1.5, 1.5],
    }
    predictions = classifier.predict(
        input_fn=lambda:mashine_data.eval_input_fn(predict_x, labels=None, batch_size=args.batch_size))

    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
    print(predictions)
    print(zip(predictions, expected))
    for predict_dict, expec in zip(predictions, expected):
        print(predict_dict)
        print(expec)
        class_id = predict_dict['class_ids'][0]
        probability = predict_dict['probabilities'][class_id]
        print(template.format(mashine_data.SPECIES[class_id], 100*probability, expec))


if __name__ == '__main__':
    tf.app.run(main)