library(ggplot2)
library(scales)

#scale_x_continuous(breaks = seq(0.1, 10000, 500))

point_size <- 3.5
line_size <- 0.5

#Statistical parity
metric      <- "statistical_parity"
file_us     <- sprintf("../pareto/adult_%s.csv", metric)
file_laftr  <- sprintf("../pareto/laftr/adult_%s.csv", metric)
file_zafar  <- sprintf("../pareto/zafar/adult_%s.csv", metric)

output_file <- sprintf("./graphs/adult_%s.pdf", metric)
df_us  <- read.csv(file_us, header=T)
df_laftr  <- read.csv(file_laftr, header=T)
df_zafar  <- read.csv(file_zafar, header=T)


ggplot() + 

geom_line(data=df_us, aes(x=error, y=unfairness,color='FairCORELS'),  size=line_size, linetype = "dashed") + 
geom_point(data=df_us, aes(x=error, y=unfairness,color='FairCORELS'), size=point_size) + 

geom_line(data=df_laftr, aes(x=error, y=unfairness,color='LAFTR'), size=line_size, linetype = "dashed") + 
geom_point(data=df_laftr, aes(x=error, y=unfairness,color='LAFTR'), size=point_size) + 

geom_line(data=df_zafar, aes(x=error, y=unfairness,color='C-LR'), size=line_size, linetype = "dashed") + 
geom_point(data=df_zafar, aes(x=error, y=unfairness,color='C-LR'), size=point_size) + 

scale_color_manual(
    values = c("FairCORELS" = "darkorange2", "LAFTR" = "darkgreen", "C-LR" = "darkred"), 
    labels = c("FairCORELS" = "FairCORELS", "LAFTR" = "LAFTR", "C-LR" = "C-LR")
) + 

theme_bw(base_size=13) + theme(legend.position = c(1, 1), legend.justification = c(1, 1), legend.background = element_blank()) + labs(color='') +

labs(x = "Error", y = expression(Delta[SP]))

ggsave(output_file, dpi=300, width=5, height=7)


#Equal opportunity
metric      <- "equal_opportunity"
file_us     <- sprintf("../pareto/adult_%s.csv", metric)
file_laftr  <- sprintf("../pareto/laftr/adult_%s.csv", metric)
file_zafar  <- sprintf("../pareto/zafar/adult_%s.csv", metric)

output_file <- sprintf("./graphs/adult_%s.pdf", metric)
df_us  <- read.csv(file_us, header=T)
df_laftr  <- read.csv(file_laftr, header=T)
df_zafar  <- read.csv(file_zafar, header=T)


ggplot() + 

geom_line(data=df_us, aes(x=error, y=unfairness,color='FairCORELS'),  size=line_size, linetype = "dashed") + 
geom_point(data=df_us, aes(x=error, y=unfairness,color='FairCORELS'), size=point_size) + 

geom_line(data=df_laftr, aes(x=error, y=unfairness,color='LAFTR'), size=line_size, linetype = "dashed") + 
geom_point(data=df_laftr, aes(x=error, y=unfairness,color='LAFTR'), size=point_size) + 

geom_line(data=df_zafar, aes(x=error, y=unfairness,color='C-LR'), size=line_size, linetype = "dashed") + 
geom_point(data=df_zafar, aes(x=error, y=unfairness,color='C-LR'), size=point_size) + 

scale_color_manual(
    values = c("FairCORELS" = "darkorange2", "LAFTR" = "darkgreen", "C-LR" = "darkred"), 
    labels = c("FairCORELS" = "FairCORELS", "LAFTR" = "LAFTR", "C-LR" = "C-LR")
) + 

theme_bw(base_size=13) + theme(legend.position = c(1, 1), legend.justification = c(1, 1), legend.background = element_blank()) + labs(color='') +

labs(x = "Error", y = expression(Delta[SP]))

ggsave(output_file, dpi=300, width=5, height=7)


#Equalized odds
metric      <- "equalized_odds"
file_us     <- sprintf("../pareto/adult_%s.csv", metric)
file_laftr  <- sprintf("../pareto/laftr/adult_%s.csv", metric)
file_zafar  <- sprintf("../pareto/zafar/adult_%s.csv", metric)

output_file <- sprintf("./graphs/adult_%s.pdf", metric)
df_us  <- read.csv(file_us, header=T)
df_laftr  <- read.csv(file_laftr, header=T)
df_zafar  <- read.csv(file_zafar, header=T)


ggplot() + 

geom_line(data=df_us, aes(x=error, y=unfairness,color='FairCORELS'),  size=line_size, linetype = "dashed") + 
geom_point(data=df_us, aes(x=error, y=unfairness,color='FairCORELS'), size=point_size) + 

geom_line(data=df_laftr, aes(x=error, y=unfairness,color='LAFTR'), size=line_size, linetype = "dashed") + 
geom_point(data=df_laftr, aes(x=error, y=unfairness,color='LAFTR'), size=point_size) + 

geom_line(data=df_zafar, aes(x=error, y=unfairness,color='C-LR'), size=line_size, linetype = "dashed") + 
geom_point(data=df_zafar, aes(x=error, y=unfairness,color='C-LR'), size=point_size) + 

scale_color_manual(
    values = c("FairCORELS" = "darkorange2", "LAFTR" = "darkgreen", "C-LR" = "darkred"), 
    labels = c("FairCORELS" = "FairCORELS", "LAFTR" = "LAFTR", "C-LR" = "C-LR")
) + 

theme_bw(base_size=13) + theme(legend.position = c(1, 1), legend.justification = c(1, 1), legend.background = element_blank()) + labs(color='') +

labs(x = "Error", y = expression(Delta[SP]))

ggsave(output_file, dpi=300, width=5, height=7)


