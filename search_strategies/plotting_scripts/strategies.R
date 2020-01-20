library(ggplot2)
library(scales)


point_size <- 1.0
line_size <- 0.1

alpha_line <- 0.5
alpha_point <- 0.9

fig_width <- 12
fig_height <- 15 



#Statistical parity
filename     <- "../pareto_merged/all.csv"
output_file <- "./graphs/strategies_without.pdf"
df  <- read.csv(filename, header=T)



ggplot() + 

geom_line(data=df, aes(x=error, y=unfairness,color=strategy),  size=line_size, linetype = "solid", alpha=alpha_line) + 
geom_point(data=df, aes(x=error, y=unfairness,color=strategy), size=point_size, alpha=alpha_point) + 

theme_bw(base_size=13) + theme(legend.position = c(1, 1), legend.justification = c(1, 1), legend.background = element_blank()) + labs(color='') +

scale_color_manual(values=c("red2", "darkorange2", "green3", "dodgerblue"))  +

labs(x = "Error", y = "Unfairness") + facet_grid(metric ~ dataset, scales="free") + theme(panel.spacing = unit(2, "lines"))

ggsave(output_file, dpi=300, width=fig_width, height=fig_height)


