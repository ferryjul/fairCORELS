library(ggplot2)
library(scales)


metric <- "statistical_parity"
with_file <- sprintf("../pareto/adult_%s_with_dem.csv", metric)
without_file <- sprintf("../pareto/adult_%s_without_dem.csv", metric)
with_laftr_file <- sprintf("../pareto/laftr/adult_%s_with_dem_laftr.csv", metric)
without_laftr_file <- sprintf("../pareto/laftr/adult_%s_without_dem_laftr.csv", metric)


output_file <- sprintf("./graphs/adult_%s.pdf", metric)


df_with  <- read.csv(with_file, header=T)
df_without  <- read.csv(without_file, header=T)
df_with_laftr  <- read.csv(with_laftr_file, header=T)
df_without_laftr  <- read.csv(without_laftr_file, header=T)


ggplot() + 
geom_line(data=df_with, aes(x=error, y=unfairness,color='FairCORELS_sen'), , size=0.2,linetype = "dashed") + 
geom_point(data=df_with, aes(x=error, y=unfairness,color='FairCORELS_sen'), size=0.9) + 

geom_line(data=df_with_laftr, aes(x=error, y=unfairness,color='LAFTR_sen'), size=0.2, linetype = "dashed") + 
geom_point(data=df_with_laftr, aes(x=error, y=unfairness,color='LAFTR_sen'), size=0.9) + 


geom_line(data=df_without, aes(x=error, y=unfairness,color='FairCORELS'),  size=0.2, linetype = "solid") + 
geom_point(data=df_without, aes(x=error, y=unfairness,color='FairCORELS'), size=0.9) + 

geom_line(data=df_without_laftr, aes(x=error, y=unfairness,color='LAFTR'), size=0.2, linetype = "solid") + 
geom_point(data=df_without_laftr, aes(x=error, y=unfairness,color='LAFTR'), size=0.9) + 

scale_color_manual(
    values = c("FairCORELS_sen" = "darkorange2", "LAFTR_sen" = "darkgreen", "FairCORELS" = "deeppink1", "LAFTR" = "blue" ), 
    labels = c("FairCORELS_sen" = expression("FairCORELS"["sen"]), "LAFTR_sen" = expression("LAFTR"["sen"]), "FairCORELS" = "FairCORELS", "LAFTR" = "LAFTR")
) + 

theme_bw(base_size=13) + theme(legend.position = c(1, 1), legend.justification = c(1, 1), legend.background = element_blank()) + labs(color='') +

labs(x = "Error", y = expression(Delta[SP]))

#scale_x_continuous(breaks = seq(0.1, 10000, 500))


ggsave(output_file, dpi=300, width=5, height=7)


metric <- "equal_opportunity"
with_file <- sprintf("../pareto/adult_%s_with_dem.csv", metric)
without_file <- sprintf("../pareto/adult_%s_without_dem.csv", metric)
with_laftr_file <- sprintf("../pareto/laftr/adult_%s_with_dem_laftr.csv", metric)
without_laftr_file <- sprintf("../pareto/laftr/adult_%s_without_dem_laftr.csv", metric)


output_file <- sprintf("./graphs/adult_%s.pdf", metric)


df_with  <- read.csv(with_file, header=T)
df_without  <- read.csv(without_file, header=T)
df_with_laftr  <- read.csv(with_laftr_file, header=T)
df_without_laftr  <- read.csv(without_laftr_file, header=T)


ggplot() + 
geom_line(data=df_with, aes(x=error, y=unfairness,color='FairCORELS_sen'), , size=0.2,linetype = "dashed") + 
geom_point(data=df_with, aes(x=error, y=unfairness,color='FairCORELS_sen'), size=0.9) + 

geom_line(data=df_with_laftr, aes(x=error, y=unfairness,color='LAFTR_sen'), size=0.2, linetype = "dashed") + 
geom_point(data=df_with_laftr, aes(x=error, y=unfairness,color='LAFTR_sen'), size=0.9) + 


geom_line(data=df_without, aes(x=error, y=unfairness,color='FairCORELS'),  size=0.2, linetype = "solid") + 
geom_point(data=df_without, aes(x=error, y=unfairness,color='FairCORELS'), size=0.9) + 

geom_line(data=df_without_laftr, aes(x=error, y=unfairness,color='LAFTR'), size=0.2, linetype = "solid") + 
geom_point(data=df_without_laftr, aes(x=error, y=unfairness,color='LAFTR'), size=0.9) + 

scale_color_manual(
    values = c("FairCORELS_sen" = "darkorange2", "LAFTR_sen" = "darkgreen", "FairCORELS" = "deeppink1", "LAFTR" = "blue" ), 
    labels = c("FairCORELS_sen" = expression("FairCORELS"["sen"]), "LAFTR_sen" = expression("LAFTR"["sen"]), "FairCORELS" = "FairCORELS", "LAFTR" = "LAFTR")
) + 

theme_bw(base_size=13) + theme(legend.position = c(1, 1), legend.justification = c(1, 1), legend.background = element_blank()) + labs(color='') +

labs(x = "Error", y = expression(Delta[EOpp]))


ggsave(output_file, dpi=300, width=5, height=7)



metric <- "equalized_odds"
with_file <- sprintf("../pareto/adult_%s_with_dem.csv", metric)
without_file <- sprintf("../pareto/adult_%s_without_dem.csv", metric)
with_laftr_file <- sprintf("../pareto/laftr/adult_%s_with_dem_laftr.csv", metric)
without_laftr_file <- sprintf("../pareto/laftr/adult_%s_without_dem_laftr.csv", metric)

output_file <- sprintf("./graphs/adult_%s.pdf", metric)


df_with  <- read.csv(with_file, header=T)
df_without  <- read.csv(without_file, header=T)
df_with_laftr  <- read.csv(with_laftr_file, header=T)
df_without_laftr  <- read.csv(without_laftr_file, header=T)


ggplot() + 
geom_line(data=df_with, aes(x=error, y=unfairness,color='FairCORELS_sen'), , size=0.2,linetype = "dashed") + 
geom_point(data=df_with, aes(x=error, y=unfairness,color='FairCORELS_sen'), size=0.9) + 

geom_line(data=df_with_laftr, aes(x=error, y=unfairness,color='LAFTR_sen'), size=0.2, linetype = "dashed") + 
geom_point(data=df_with_laftr, aes(x=error, y=unfairness,color='LAFTR_sen'), size=0.9) + 


geom_line(data=df_without, aes(x=error, y=unfairness,color='FairCORELS'),  size=0.2, linetype = "solid") + 
geom_point(data=df_without, aes(x=error, y=unfairness,color='FairCORELS'), size=0.9) + 

geom_line(data=df_without_laftr, aes(x=error, y=unfairness,color='LAFTR'), size=0.2, linetype = "solid") + 
geom_point(data=df_without_laftr, aes(x=error, y=unfairness,color='LAFTR'), size=0.9) + 

scale_color_manual(
    values = c("FairCORELS_sen" = "darkorange2", "LAFTR_sen" = "darkgreen", "FairCORELS" = "deeppink1", "LAFTR" = "blue" ), 
    labels = c("FairCORELS_sen" = expression("FairCORELS"["sen"]), "LAFTR_sen" = expression("LAFTR"["sen"]), "FairCORELS" = "FairCORELS", "LAFTR" = "LAFTR")
) + 

theme_bw(base_size=13) + theme(legend.position = c(1, 1), legend.justification = c(1, 1), legend.background = element_blank()) + labs(color='') +

labs(x = "Error", y = expression(Delta[EOdds]))


ggsave(output_file, dpi=300, width=5, height=7)


