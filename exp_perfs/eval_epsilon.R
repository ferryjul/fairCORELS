library(readr)
library(dplyr)
library(ggplot2)
library(scales)
library(grid)
library(RColorBrewer)
library(digest)
library(readr)
library(stringr)

#1---------------------------------------------
df  <- read.csv("output/df_epsilon_fairness_1.csv", header=T)

ggplot(df, aes(x = factor(parameter), y = accuracy, fill = factor(parameter))) +

  geom_boxplot(alpha = 0.80) +
  geom_point(aes(color=factor(fold_id))) +

  labs(x = expression(epsilon), y = "Accuracy", title="Statisitcal parity") +
  theme_bw(base_size=13) + theme(legend.position="none")
ggsave("graphs/df_epsilon_accuracy_1.png", dpi=300, width=10, height=5)

ggplot(df, aes(x = factor(parameter), y = unfairness, fill = factor(parameter))) +

  geom_boxplot(alpha = 0.80) +
  geom_point(aes(color=factor(fold_id))) +

  labs(x = expression(epsilon), y = "Unfainess", title="Statisitcal parity") +
  theme_bw(base_size=13) + theme(legend.position="none")

ggsave("graphs/df_epsilon_unfairness_1.png", dpi=300, width=10, height=5)


#2---------------------------------------------

df  <- read.csv("output/df_epsilon_fairness_2.csv", header=T)

ggplot(df, aes(x = factor(parameter), y = accuracy, fill = factor(parameter))) +

  geom_boxplot(alpha = 0.80) +
  geom_point(aes(color=factor(fold_id))) +

  labs(x = expression(epsilon), y = "Accuracy", title="Predictive parity") +
  theme_bw(base_size=13) + theme(legend.position="none")

ggsave("graphs/df_epsilon_accuracy_2.png", dpi=300, width=10, height=5)

ggplot(df, aes(x = factor(parameter), y = unfairness, fill = factor(parameter))) +

  geom_boxplot(alpha = 0.80) +
  geom_point(aes(color=factor(fold_id))) +

  labs(x = expression(epsilon), y = "Unfainess", title="Predictive parity") +
  theme_bw(base_size=13) + theme(legend.position="none")

ggsave("graphs/df_epsilon_unfairness_2.png", dpi=300, width=10, height=5)


#3---------------------------------------------

df  <- read.csv("output/df_epsilon_fairness_3.csv", header=T)

ggplot(df, aes(x = factor(parameter), y = accuracy, fill = factor(parameter))) +

  geom_boxplot(alpha = 0.80) +
  geom_point(aes(color=factor(fold_id))) +

  labs(x = expression(epsilon), y = "Accuracy", title="Predictive equality") +
  theme_bw(base_size=13) + theme(legend.position="none")

ggsave("graphs/df_epsilon_accuracy_3.png", dpi=300, width=10, height=5)

ggplot(df, aes(x = factor(parameter), y = unfairness, fill = factor(parameter))) +

  geom_boxplot(alpha = 0.80) +
  geom_point(aes(color=factor(fold_id))) +

  labs(x = expression(epsilon), y = "Unfainess", title="Predictive equality") +
  theme_bw(base_size=13) + theme(legend.position="none")

ggsave("graphs/df_epsilon_unfairness_3.png", dpi=300, width=10, height=5)

#4---------------------------------------------

df  <- read.csv("output/df_epsilon_fairness_4.csv", header=T)

ggplot(df, aes(x = factor(parameter), y = accuracy, fill = factor(parameter))) +

  geom_boxplot(alpha = 0.80) +
  geom_point(aes(color=factor(fold_id))) +

  labs(x = expression(epsilon), y = "Accuracy", title="Equal opportunity") +
  theme_bw(base_size=13) + theme(legend.position="none")

ggsave("graphs/df_epsilon_accuracy_4.png", dpi=300, width=10, height=5)

ggplot(df, aes(x = factor(parameter), y = unfairness, fill = factor(parameter))) +

  geom_boxplot(alpha = 0.80) +
  geom_point(aes(color=factor(fold_id))) +

  labs(x = expression(epsilon), y = "Unfainess", title="Equal opportunity") +
  theme_bw(base_size=13) + theme(legend.position="none")

ggsave("graphs/df_epsilon_unfairness_4.png", dpi=300, width=10, height=5)