library(ggplot2)
library(scales)

#scale_x_continuous(breaks = seq(0.1, 10000, 500))

point_size <- 2.5
line_size <- 0.2

alpha_line <- 0.5
alpha_point <- 1.0

fig_width <- 5
fig_height <- 4

basesize <- 20


#Statistical parity
metric      <- "statistical_parity"
file_us     <- sprintf("../pareto/faircorels/compas_without/%s.csv", metric)
file_laftr  <- sprintf("../pareto/laftr/compas_without/%s.csv", metric)
file_zafar  <- sprintf("../pareto/zafar/compas_without/%s.csv", metric)

output_file <- sprintf("./graphs/compas_without_%s.pdf", metric)
df_us  <- read.csv(file_us, header=T)
df_laftr  <- read.csv(file_laftr, header=T)
df_zafar  <- read.csv(file_zafar, header=T)


ggplot() + 

geom_line(data=df_us, aes(x=error, y=unfairness,color='FairCORELS'),  size=line_size, linetype = "dashed", alpha=alpha_line) + 
geom_point(data=df_us, aes(x=error, y=unfairness,color='FairCORELS',shape='FairCORELS'), size=point_size, alpha=alpha_point) + 

geom_line(data=df_laftr, aes(x=error, y=unfairness,color='LAFTR'), size=line_size, linetype = "dashed", alpha=alpha_line) + 
geom_point(data=df_laftr, aes(x=error, y=unfairness,color='LAFTR',shape='LAFTR'), size=point_size, alpha=alpha_point) + 

geom_line(data=df_zafar, aes(x=error, y=unfairness,color='C-LR'), size=line_size, linetype = "dashed", alpha=alpha_line) + 
geom_point(data=df_zafar, aes(x=error, y=unfairness,color='C-LR',shape='C-LR'), size=point_size, alpha=alpha_point) + 

scale_color_manual(
    name="Methods",
    values = c("FairCORELS" = "darkorange2", "LAFTR" = "darkgreen", "C-LR" = "darkred"), 
    labels = c("FairCORELS" = "FairCORELS", "LAFTR" = "LAFTR", "C-LR" = "C-LR")
) + 

scale_shape_manual(
    name="Methods",
    values = c("FairCORELS" = 0, "LAFTR" = 1, "C-LR" = 2), 
    labels = c("FairCORELS" = "FairCORELS", "LAFTR" = "LAFTR", "C-LR" = "C-LR")
) + 

theme_bw(base_size=basesize) + theme(legend.position = c(1, 1), legend.justification = c(1, 1), legend.background = element_blank()) + labs(color='') +

labs(x = "Error", y = expression(Delta[SP]))

ggsave(output_file, dpi=300, width=fig_width, height=fig_height)


#Equal opportunity
metric      <- "equal_opportunity"
file_us     <- sprintf("../pareto/faircorels/compas_without/%s.csv", metric)
file_laftr  <- sprintf("../pareto/laftr/compas_without/%s.csv", metric)
file_zafar  <- sprintf("../pareto/zafar/compas_without/%s.csv", metric)

output_file <- sprintf("./graphs/compas_without_%s.pdf", metric)
df_us  <- read.csv(file_us, header=T)
df_laftr  <- read.csv(file_laftr, header=T)
df_zafar  <- read.csv(file_zafar, header=T)

ggplot() + 

geom_line(data=df_us, aes(x=error, y=unfairness,color='FairCORELS'),  size=line_size, linetype = "dashed", alpha=alpha_line) + 
geom_point(data=df_us, aes(x=error, y=unfairness,color='FairCORELS',shape='FairCORELS'), size=point_size, alpha=alpha_point) + 

geom_line(data=df_laftr, aes(x=error, y=unfairness,color='LAFTR'), size=line_size, linetype = "dashed", alpha=alpha_line) + 
geom_point(data=df_laftr, aes(x=error, y=unfairness,color='LAFTR',shape='LAFTR'), size=point_size, alpha=alpha_point) + 

geom_line(data=df_zafar, aes(x=error, y=unfairness,color='C-LR'), size=line_size, linetype = "dashed", alpha=alpha_line) + 
geom_point(data=df_zafar, aes(x=error, y=unfairness,color='C-LR',shape='C-LR'), size=point_size, alpha=alpha_point) + 

scale_color_manual(
    name="Methods",
    values = c("FairCORELS" = "darkorange2", "LAFTR" = "darkgreen", "C-LR" = "darkred"), 
    labels = c("FairCORELS" = "FairCORELS", "LAFTR" = "LAFTR", "C-LR" = "C-LR")
) + 

scale_shape_manual(
    name="Methods",
    values = c("FairCORELS" = 0, "LAFTR" = 1, "C-LR" = 2), 
    labels = c("FairCORELS" = "FairCORELS", "LAFTR" = "LAFTR", "C-LR" = "C-LR")
) + 

theme_bw(base_size=basesize) + theme(legend.position = c(1, 1), legend.justification = c(1, 1), legend.background = element_blank()) + labs(color='') +

labs(x = "Error", y = expression(Delta[EOpp]))

ggsave(output_file, dpi=300, width=fig_width, height=fig_height)


#Equalized odds
metric      <- "equalized_odds"
file_us     <- sprintf("../pareto/faircorels/compas_without/%s.csv", metric)
file_laftr  <- sprintf("../pareto/laftr/compas_without/%s.csv", metric)
file_zafar  <- sprintf("../pareto/zafar/compas_without/%s.csv", metric)

output_file <- sprintf("./graphs/compas_without_%s.pdf", metric)
df_us  <- read.csv(file_us, header=T)
df_laftr  <- read.csv(file_laftr, header=T)
df_zafar  <- read.csv(file_zafar, header=T)


ggplot() + 

geom_line(data=df_us, aes(x=error, y=unfairness,color='FairCORELS'),  size=line_size, linetype = "dashed", alpha=alpha_line) + 
geom_point(data=df_us, aes(x=error, y=unfairness,color='FairCORELS',shape='FairCORELS'), size=point_size, alpha=alpha_point) + 

geom_line(data=df_laftr, aes(x=error, y=unfairness,color='LAFTR'), size=line_size, linetype = "dashed", alpha=alpha_line) + 
geom_point(data=df_laftr, aes(x=error, y=unfairness,color='LAFTR',shape='LAFTR'), size=point_size, alpha=alpha_point) + 

geom_line(data=df_zafar, aes(x=error, y=unfairness,color='C-LR'), size=line_size, linetype = "dashed", alpha=alpha_line) + 
geom_point(data=df_zafar, aes(x=error, y=unfairness,color='C-LR',shape='C-LR'), size=point_size, alpha=alpha_point) + 

scale_color_manual(
    name="Methods",
    values = c("FairCORELS" = "darkorange2", "LAFTR" = "darkgreen", "C-LR" = "darkred"), 
    labels = c("FairCORELS" = "FairCORELS", "LAFTR" = "LAFTR", "C-LR" = "C-LR")
) + 

scale_shape_manual(
    name="Methods",
    values = c("FairCORELS" = 0, "LAFTR" = 1, "C-LR" = 2), 
    labels = c("FairCORELS" = "FairCORELS", "LAFTR" = "LAFTR", "C-LR" = "C-LR")
) + 

theme_bw(base_size=basesize) + theme(legend.position = c(1, 1), legend.justification = c(1, 1), legend.background = element_blank()) + labs(color='') +

labs(x = "Error", y = expression(Delta[EOdds]))

ggsave(output_file, dpi=300, width=fig_width, height=fig_height)


