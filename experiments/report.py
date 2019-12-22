import pandas as pd


import argparse


# dict 

dataset_dict = {
    1 : 'adult',
    2 : 'compas',
    3 : 'german_credit',
    4 : 'default_credit',
    5 : 'adult_gender',
    6 : 'adult_no_relationship'
}

metric_dict = {
    1 : 'statistical_parity',
    2 : 'predictive_parity',
    3 : 'predictive_equality',
    4 : 'equal_opportunity',
    5 : 'equalized_odds',
    6 : 'conditional_use_accuracy_equality'
}

suffix = {
    1 : 'without_dem',
    2 : 'with_dem',
}

# parser initialization
parser = argparse.ArgumentParser(description='Reporting of FairCORELS results')
parser.add_argument('--id', type=int, default=1, help='dataset id: 1-4')
parser.add_argument('--m', type=int, default=1, help='fairness metric: 1-6')
parser.add_argument('--attr', type=int, default=1, help='use sensitive attribute: 1 no, 2 yes')

parser.add_argument('--exp', type=str, default="results_1M", help='use sensitive attribute: 1 no, 2 yes')


args = parser.parse_args()


print("=====>>>>>>>> Report for experiments {} on {} | {} | {}".format(args.exp, 
                                                                      dataset_dict[args.id], 
                                                                      metric_dict[args.m], 
                                                                      suffix[args.attr]))


input_file='./{}/{}_{}_{}.csv'.format(args.exp, dataset_dict[args.id], metric_dict[args.m], suffix[args.attr])

df = pd.read_csv(input_file)

print('===='*28)

def best_accuracy(df, at):
    accuracy = df.accuracy.tolist()
    unfairness = df.unfairness.tolist()
    description = df.models.tolist()



    models  = list(zip(accuracy, unfairness, description))

    
    models = filter(lambda x: x[1] <= at, models)
    model = sorted(models, key=lambda x: x[0], reverse=True)

    model = 'acc= {}, unf= {}'.format(model[0][0], model[0][1]) if len(model) > 0 else None

    print('===========>>> Best model at unfairness <= {} : {}'.format(at, model))




best_accuracy(df,1.0)
best_accuracy(df,0.05)
best_accuracy(df,0.04)
best_accuracy(df,0.03)
best_accuracy(df,0.02)
best_accuracy(df,0.01)



