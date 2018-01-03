import time
import pandas as pd


contest_basic_train = pd.read_csv('./data/contest_basic_train.tsv', sep='\t', header=0)

contest_basic_train['BORN_PROVINCE'] = [card[:2] for card in contest_basic_train['ID_CARD']]  # TODO: replace as 'card[:4]' or 'card[:6]'
contest_basic_train['SEX'] = [int(card[-2]) % 2 for card in contest_basic_train['ID_CARD']]
contest_basic_train = contest_basic_train.drop(['ID_CARD'], axis=1)

contest_basic_train['LOAN_DATE'] = [int(time.mktime(time.strptime(date, "%Y/%m/%d"))) for date in contest_basic_train['LOAN_DATE']]

contest_basic_train['WORK_PROVINCE'] = [province // 1e+4 for province in contest_basic_train['WORK_PROVINCE']]  # TODO: replace as '// 1e+2' or '// 1'

dummy_fields = ['BORN_PROVINCE', 'SEX', 'AGENT', 'IS_LOCAL', 'WORK_PROVINCE', 'EDU_LEVEL', 'MARRY_STATUS']
for each in dummy_fields:
    dummies = pd.get_dummies(contest_basic_train[each], dummy_na=True, prefix=each, drop_first=False)
    contest_basic_train = pd.concat([contest_basic_train, dummies], axis=1)
contest_basic_train = contest_basic_train.drop(dummy_fields, axis=1)

# quant_features = ['SALARY']
# scaled_features = {}
# for each in quant_features:
#     mean, std = contest_basic_train[each].mean(), contest_basic_train[each].std()
#     scaled_features[each] = [mean, std]
#     contest_basic_train.loc[:, each] = (contest_basic_train[each] - mean)/std

print(contest_basic_train.head(3))
for key in contest_basic_train.keys():
    print(key)
