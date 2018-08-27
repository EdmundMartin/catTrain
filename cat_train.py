from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split
import numpy as np


class CatTrainer:

    def __init__(self, train_df):
        self.train_df = train_df
        self.model = None
        self.X = None
        self.y = None
        self.categorical_features_indices = None

    def _replace_null_values(self, value, inplace=True):
        self.train_df.fillna(value, inplace=inplace)

    def _default_args_or_kwargs(self, **kwargs):
        params = {
            'iterations': 500,
            'learning_rate': 0.1,
            'eval_metric': 'Accuracy',
            'random_seed': 42,
            'logging_level': 'Silent',
            'use_best_model': True
        }
        for k in kwargs.keys():
            if k in params:
                params[k] = kwargs.get(k)
        return params

    def prepare_x_y(self, label, null_value=-999):
        self._replace_null_values(null_value)
        self.X = self.train_df.drop(label, axis=1)
        self.y = self.train_df[label]

    def create_model(self, **kwargs):
        params = self._default_args_or_kwargs(**kwargs)
        if not self.model:
            self.model = CatBoostClassifier(**params)
        else:
            raise ValueError("Cannot overwrite existing model")

    def train_model(self, train_size=0.75, random_state=42, **kwargs):
        X_train, X_validation, y_train, y_validation = train_test_split(self.X, self.y, train_size=train_size,
                                                                        random_state=random_state)
        if not self.categorical_features_indices:
            self.categorical_features_indices = np.where(self.X.dtypes != np.float)[0]
        if not self.model:
            self.create_model(**kwargs)
        self.model.fit(
            X_train, y_train,
            cat_features=self.categorical_features_indices,
            eval_set=(X_validation, y_validation),
            logging_level='Verbose',
        )

    def model_cross_validation(self):
        cv_data = cv(
            Pool(self.X, self.y, cat_features=self.categorical_features_indices),
            self.model.get_params()
        )
        return np.max(cv_data['test-Accuracy-mean'])

    def save_model(self, name):
        self.model.save_model('{}.dump'.format(name))

    def load_model(self, name):
        self.model.load_model('{}.dump'.format(name))

    def predict(self, dataframe, null_value=999, inplace=True):
        dataframe.fillna(null_value, inplace=inplace)
        results = self.model.predict(dataframe)
        return results


if __name__ == '__main__':
    from catboost.datasets import titanic
    train_df, test_df = titanic()
    c = CatTrainer(train_df)
    c.prepare_x_y('Survived')
    print(c.X, c.y)
    c.train_model()
    score = c.model_cross_validation()
    print(score)
    c.save_model('demo')
    res = c.predict(test_df)
    print(res)