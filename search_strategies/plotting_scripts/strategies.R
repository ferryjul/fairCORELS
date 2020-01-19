library(ggplot2)
library(scales)


point_size <- 3.5
line_size <- 1.0

alpha_line <- 1.0
alpha_point <- 0.7

fig_width <- 8
fig_height <- 15 



#Statistical parity
filename     <- "../pareto_merged/all.csv"
output_file <- "./graphs/compare.png"
df  <- read.csv(filename, header=T)



ggplot() + 

geom_line(data=df, aes(x=error, y=unfairness,color=strategy),  size=line_size, linetype = "solid", alpha=alpha_line) + 
#geom_point(data=df, aes(x=error, y=unfairness,color=strategy), size=point_size, alpha=alpha_point) + 

theme_bw(base_size=13) + theme(legend.position = c(1, 1), legend.justification = c(1, 1), legend.background = element_blank()) + labs(color='') +

labs(x = "Error", y = "Unfairness") + facet_grid(metric ~ dataset)

ggsave(output_file, dpi=300, width=fig_width, height=fig_height)


