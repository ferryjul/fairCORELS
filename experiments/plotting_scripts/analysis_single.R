library(ggplot2)
library(scales)
library(argparse)


library(dict)

datasets <- dict()
metrics <- dict()
suffix <- dict()

datasets[[1]] <- "adult"
datasets[[2]] <- "compas"
datasets[[3]] <- "german_credit"
datasets[[4]] <- "default_credit"


metrics[[1]] <- "statistical_parity"
metrics[[2]] <- "predictive_parity"
metrics[[3]] <- "predictive_equality"
metrics[[4]] <- "equal_opportunity"
metrics[[5]] <- "equalized_odds"
metrics[[6]] <- "conditional_use_accuracy_equality"


suffix[[1]] <- "without_dem"
suffix[[2]] <- "with_dem"

parser <- ArgumentParser()
parser$add_argument("--id", default=1, type="integer", help="dataset id: 1-4")
parser$add_argument("--m", default=1, type="integer", help="fairness metric 1-6")
parser$add_argument("--attr", default=1, type="integer", help="use sensitive attr 1-2")
parser$add_argument("--exp", default='results', type="character", help="experiment folder")



args <- parser$parse_args()

input_file <- sprintf("./data/%s_%s_%s_%s.csv", args$exp, datasets[[args$id]], metrics[[args$m]], suffix[[args$attr]])
output_file <- sprintf("./graphs/%s_%s_%s_%s.png", args$exp, datasets[[args$id]], metrics[[args$m]], suffix[[args$attr]])


df  <- read.csv(input_file, header=T)

ggplot(df, aes(x=error, y=unfairness)) + 
  geom_line(size=0.1) +
  geom_point(size=0.5) +
  labs(x = "error", y = "unfairness") +
  theme_bw(base_size=13)

ggsave(output_file, dpi=300, width=6, height=10)
