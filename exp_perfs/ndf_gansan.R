library(readr)
library(dplyr)
library(ggplot2)
library(scales)
library(grid)
library(RColorBrewer)
library(digest)
library(readr)
library(stringr)


df  <- read.csv("data/ndf_gansan_sacc.csv", header=T)

ggplot(df, aes(x=fidelity, y=fairness)) + 
    geom_point(aes(color=alpha, shape=epoch)) +
    geom_line(size=0.05) +
    labs(x = "Fidelity", y = "Fairness(Sacc)", color = expression(alpha)) +
    theme_bw(base_size=13)

ggsave("img/pareto_sacc.pdf", dpi=10, width=10, height=5)


df  <- read.csv("data/ndf_gansan_ber.csv", header=T)

ggplot(df, aes(x=fidelity, y=fairness)) + 
    geom_point(aes(color=alpha, shape=epoch)) +
    geom_line(size=0.05) +
    labs(x = "Fidelity", y = "Fairness(BER)", color = expression(alpha)) +
    theme_bw(base_size=13)

ggsave("img/pareto_ber.pdf", dpi=10, width=10, height=5)


