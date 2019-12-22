library(ggplot2)
library(scales)
library(argparse)


library(dict)

datasets <- dict()
metrics <- dict()

datasets[[1]] <- "adult"
datasets[[2]] <- "compas"
datasets[[3]] <- "german_credit"
datasets[[4]] <- "default_credit"
datasets[[5]] <- "adult_gender"
datasets[[6]] <- "adult_no_relationship"


metrics[[1]] <- "statistical_parity"
metrics[[2]] <- "predictive_parity"
metrics[[3]] <- "predictive_equality"
metrics[[4]] <- "equal_opportunity"
metrics[[5]] <- "equalized_odds"
metrics[[6]] <- "conditional_use_accuracy_equality"


parser <- ArgumentParser()
parser$add_argument("--id", default=1, type="integer", help="dataset id: 1-4")
parser$add_argument("--m", default=1, type="integer", help="fairness metric 1-6")
parser$add_argument("--exp", default='results', type="character", help="experiment folder")


args <- parser$parse_args()

input_file <- sprintf("./data/%s_%s_%s_with_dem.csv", args$exp, datasets[[args$id]], metrics[[args$m]])
input_file2 <- sprintf("./data/%s_%s_%s_without_dem.csv", args$exp, datasets[[args$id]], metrics[[args$m]])

output_file <- sprintf("./graphs/%s_%s_%s.png", args$exp, datasets[[args$id]], metrics[[args$m]])


df  <- read.csv(input_file, header=T)
df2  <- read.csv(input_file2, header=T)

ggplot() + 
geom_line(data=df, aes(x=error, y=unfairness), color='red', size=0.1) + 
geom_point(data=df, aes(x=error, y=unfairness), color='red', size=0.5) + 
geom_line(data=df2, aes(x=error, y=unfairness), color='green', size=0.1) + 
geom_point(data=df2, aes(x=error, y=unfairness), color='green', size=0.5) + 
theme_bw(base_size=13)


ggsave(output_file, dpi=300, width=6, height=10)
