library(ggplot2)
library(scales)


point_size <- 1.5
line_size <- 0.5

alpha_line <- 0.4
alpha_point <- 1.0

fig_width <- 5
fig_height <- 4



#####Adult

dataset      <- "COMPAS"

#Statistical parity
metric      <- "statistical_parity"
filename     <- sprintf("../pareto_merged_ulb/%s_%s.csv", dataset, metric)
output_file <- sprintf("./graphs/ulb_%s_%s.pdf", dataset, metric)

df  <- read.csv(filename, header=T)


ggplot() + 

geom_line(data=df, aes(x=error, y=unfairness,color=strategy),  size=line_size, linetype = "dashed", alpha=alpha_line) + 
geom_point(data=df, aes(x=error, y=unfairness,color=strategy), size=point_size, alpha=alpha_point) + 

theme_bw(base_size=13) + theme(legend.position = c(1, 1), legend.justification = c(1, 1), legend.background = element_blank()) + labs(color='') +

scale_color_manual(values=c("red2", "dodgerblue"))  +

labs(x = "Error", y = "Unfairness") 

ggsave(output_file, dpi=300, width=fig_width, height=fig_height)


#Predictive parity
metric      <- "predictive_parity"
filename     <- sprintf("../pareto_merged_ulb/%s_%s.csv", dataset, metric)
output_file <- sprintf("./graphs/ulb_%s_%s.pdf", dataset, metric)

df  <- read.csv(filename, header=T)


ggplot() + 

geom_line(data=df, aes(x=error, y=unfairness,color=strategy),  size=line_size, linetype = "dashed", alpha=alpha_line) + 
geom_point(data=df, aes(x=error, y=unfairness,color=strategy), size=point_size, alpha=alpha_point) + 

theme_bw(base_size=13) + theme(legend.position = c(1, 1), legend.justification = c(1, 1), legend.background = element_blank()) + labs(color='') +

scale_color_manual(values=c("red2", "dodgerblue"))  +

labs(x = "Error", y = "Unfairness") 

ggsave(output_file, dpi=300, width=fig_width, height=fig_height)


#Predictive equality
metric      <- "predictive_equality"
filename     <- sprintf("../pareto_merged_ulb/%s_%s.csv", dataset, metric)
output_file <- sprintf("./graphs/ulb_%s_%s.pdf", dataset, metric)

df  <- read.csv(filename, header=T)


ggplot() + 

geom_line(data=df, aes(x=error, y=unfairness,color=strategy),  size=line_size, linetype = "dashed", alpha=alpha_line) + 
geom_point(data=df, aes(x=error, y=unfairness,color=strategy), size=point_size, alpha=alpha_point) + 

theme_bw(base_size=13) + theme(legend.position = c(1, 1), legend.justification = c(1, 1), legend.background = element_blank()) + labs(color='') +

scale_color_manual(values=c("red2", "dodgerblue"))  +

labs(x = "Error", y = "Unfairness") 

ggsave(output_file, dpi=300, width=fig_width, height=fig_height)


#Equal opportunity
metric      <- "equal_opportunity"
filename     <- sprintf("../pareto_merged_ulb/%s_%s.csv", dataset, metric)
output_file <- sprintf("./graphs/ulb_%s_%s.pdf", dataset, metric)

df  <- read.csv(filename, header=T)


ggplot() + 

geom_line(data=df, aes(x=error, y=unfairness,color=strategy),  size=line_size, linetype = "dashed", alpha=alpha_line) + 
geom_point(data=df, aes(x=error, y=unfairness,color=strategy), size=point_size, alpha=alpha_point) + 

theme_bw(base_size=13) + theme(legend.position = c(1, 1), legend.justification = c(1, 1), legend.background = element_blank()) + labs(color='') +

scale_color_manual(values=c("red2", "dodgerblue"))  +

labs(x = "Error", y = "Unfairness") 

ggsave(output_file, dpi=300, width=fig_width, height=fig_height)


#Equalized odds
metric      <- "equalized_odds"
filename     <- sprintf("../pareto_merged_ulb/%s_%s.csv", dataset, metric)
output_file <- sprintf("./graphs/ulb_%s_%s.pdf", dataset, metric)

df  <- read.csv(filename, header=T)


ggplot() + 

geom_line(data=df, aes(x=error, y=unfairness,color=strategy),  size=line_size, linetype = "dashed", alpha=alpha_line) + 
geom_point(data=df, aes(x=error, y=unfairness,color=strategy), size=point_size, alpha=alpha_point) + 

theme_bw(base_size=13) + theme(legend.position = c(1, 1), legend.justification = c(1, 1), legend.background = element_blank()) + labs(color='') +

scale_color_manual(values=c("red2", "dodgerblue"))  +

labs(x = "Error", y = "Unfairness") 

ggsave(output_file, dpi=300, width=fig_width, height=fig_height)


#Conditional use accuracy equality
metric      <- "conditional_use_accuracy_equality"
filename     <- sprintf("../pareto_merged_ulb/%s_%s.csv", dataset, metric)
output_file <- sprintf("./graphs/ulb_%s_%s.pdf", dataset, metric)

df  <- read.csv(filename, header=T)


ggplot() + 

geom_line(data=df, aes(x=error, y=unfairness,color=strategy),  size=line_size, linetype = "dashed", alpha=alpha_line) + 
geom_point(data=df, aes(x=error, y=unfairness,color=strategy), size=point_size, alpha=alpha_point) + 

theme_bw(base_size=13) + theme(legend.position = c(1, 1), legend.justification = c(1, 1), legend.background = element_blank()) + labs(color='') +

scale_color_manual(values=c("red2", "dodgerblue"))  +

labs(x = "Error", y = "Unfairness") 

ggsave(output_file, dpi=300, width=fig_width, height=fig_height)




