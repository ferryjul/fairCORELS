library(ggplot2)
library(scales)


point_size <- 1.0
line_size <- 0.1

alpha_line <- 0.5
alpha_point <- 1.0

fig_width <- 10
fig_height <- 18



filename     <- "../pareto_merged/adult.csv"
output_file <- "./graphs/ulb_adult.pdf"
df  <- read.csv(filename, header=T)



ggplot() + 

geom_line(data=df, aes(x=error, y=unfairness,color=strategy),  size=line_size, linetype = "dashed", alpha=alpha_line) + 
geom_point(data=df, aes(x=error, y=unfairness,color=strategy,shape=strategy), size=point_size, alpha=alpha_point) + 

theme_bw(base_size=13) + theme(legend.position = c(1, 1), legend.justification = c(1, 1), legend.background = element_blank()) + labs(color='') +


scale_color_manual(
    name="Methods",
    values=c("red2", "darkorange2", "green3", "dodgerblue")
) + 

scale_shape_manual(
    name="Methods",
    values = c(0, 1, 2, 3)
) + 

labs(x = "Error", y = "Unfairness") + facet_grid(metric ~ dataset, scales="free") + theme(panel.spacing = unit(1.2, "lines"))

ggsave(output_file, dpi=600, width=fig_width, height=fig_height)


