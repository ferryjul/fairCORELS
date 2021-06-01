import pandas as pd 
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

def proc_number(n):
    return round(n, 3)


plotModesList = ['unf_perf_gene', 'unf_gene', 'acc_perf_gene', 'training_acc', 'acc_gene', 'test_acc', 'unf_violation', 'summary_table']
parser = argparse.ArgumentParser(description='Analysis of FairCORELS results')
parser.add_argument('--dataset', type=str, default="compas", help='either adult or compas')
args = parser.parse_args()
full = 2
policy = "bfs"
if args.dataset == "compas":
    max_time=1200
elif args.dataset == "german_credit":
    max_time=2400
else:
    print("Unknown dataset. Exiting.")
    exit()
metrics=[1,3,4,5]
metricsNames = ["SP", "PE", "EO", "EOdds"]
lines = []
for m in metrics:
    fileName = './figures/figures_paper/table_learningQ_dataset%s_metric%d_policy%s_maxTime%d.csv' %(args.dataset, m, policy, max_time)
    fileContent = pd.read_csv(fileName)
    lines.append(fileContent.values[0])
fileName = './figures/figures_paper/table_learningQ_dataset%s_SUMMARY_policy%s_maxTime%d.csv' %(args.dataset, policy, max_time)
for i, line in enumerate(lines):
    for j, el in enumerate(line):
        try:
            line[j] = np.round(el, 3)
        except:
            mm=1
    lines[i] = line
with open(fileName, mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    if full == 1:
        csv_writer.writerow(['min_epsilon(total_vals)', 'test_acc_percentbest(average)', 'test_unf_violation(average)', 'test_unf_violation(median)', 'test_acc_percentbest(average)', 'test_unf_violation(average)', 'test_unf_violation(median)', 'test_acc_percentbest(average)', 'test_unf_violation(average)', 'test_unf_violation(median)'])
    elif full == 0:
        csv_writer.writerow(['min_epsilon(total_vals)', 'test_acc_percentbest(average)', 'test_unf_violation(average)', 'test_acc_percentbest(average)', 'test_unf_violation(average)', 'test_acc_percentbest(average)', 'test_unf_violation(average)'])
    elif full == 2:
        csv_writer.writerow(['min_epsilon(total_vals)', 'train_acc_percentbest(average)', 'test_acc_percentbest(average)', 'test_unf_violation(average)', 'train_acc_percentbest(average)', 'test_acc_percentbest(average)', 'test_unf_violation(average)', 'train_acc_percentbest(average)', 'test_acc_percentbest(average)', 'test_unf_violation(average)'])        
    for i, line in enumerate(lines):
        csv_writer.writerow(metricsNames[i])
        csv_writer.writerow(line)
    exit()