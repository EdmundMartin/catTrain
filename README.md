# catTrain
Using Yandex's Catboost library to train models on categorical data

## Using with Titanic Dataset
```python3
 from catboost.datasets import titanic
 
 train_df, test_df = titanic()
 c = CatTrainer(train_df, test_df)
 
 # Set the DF column which contains the label
 c.prepare_x_y('Survived')
 
 c.train_model()
 
 accuracy_score = c.model_cross_validation()
 print(accuracy_score)
  c.save_model('demo')
```
