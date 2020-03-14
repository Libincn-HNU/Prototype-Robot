import matchzoo as mz
import pandas as pd
print(mz.__version__)
data_pack = mz.pack(pd.read_csv('match-zoo-corpus-top-1w.csv'))
print(data_pack[-10:])

data_pack.relation['label'] = data_pack.relation['label'].astype('float32')
frame = data_pack.frame

task = mz.tasks.Ranking()
train_raw = data_pack # mz.datasets.toy.load_data(stage='train', task=task)
test_raw = data_pack # mz.datasets.toy.load_data(stage='test', task=task)

model = mz.load_model('step-2-mz-model')

preprocessor = mz.preprocessors.BasicPreprocessor()
preprocessor.fit(train_raw, verbose=0)  ## init preprocessor inner state.
# train_processed = preprocessor.transform(train_raw, verbose=5)
test_processed = preprocessor.transform(test_raw, verbose=0)

# x, y = train_processed.unpack()
test_x, test_y = test_processed.unpack()

results = model.predict(test_x)

print(type(results))
print(len(results))
print(results)
for idx, item in enumerate(results[:20]):
    print('*'*100)
    print(idx)
    print(item)
for idx, item in enumerate(results[-20:]):
    print('*'*100)
    print(idx)
    print(item)
