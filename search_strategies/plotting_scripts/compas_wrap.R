library(ggplot2)
library(scales)


point_size <- 2.5
line_size <- 0.2

alpha_line <- 0.5
alpha_point <- 1.0

fig_width <- 30
fig_height <- 6



filename     <- "../pareto_merged/compas_wrap.csv"
output_file <- "./graphs/strategies_compas.pdf"
df  <- read.csv(filename, header=T)



ggplot() + 

geom_line(data=df, aes(x=error, y=unfairness,color=strategy),  size=line_size, linetype = "dashed", alpha=alpha_line) + 
geom_point(data=df, aes(x=error, y=unfairness,color=strategy,shape=strategy), size=point_size, alpha=alpha_point) + 

theme_bw(base_size=20) + theme(legend.position = c(1, 1), legend.justification = c(1, 1), legend.background = element_blank()) + labs(color='') +


scale_color_manual(
    name="Search strategies",
    values=c("red2", "darkorange2", "green3", "dodgerblue")
) + 

scale_shape_manual(
    name="Search strategies",
    values = c(0, 1, 2, 3)
) + 

labs(x = "Error", y = "Unfairness") + facet_wrap(~metric, scales="free", ncol = 6) + theme(panel.spacing = unit(1.2, "lines"))

ggsave(output_file, dpi=300, width=fig_width, height=fig_height)


