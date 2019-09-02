library(readr)
library(dplyr)
library(ggplot2)
library(scales)
library(grid)
library(RColorBrewer)
library(digest)
library(readr)
library(stringr)


df_adult  <- read.csv("data/adult_fine.csv", header=T)
ggplot(df_adult, aes(x=Error, y=Unfairness)) + 
  geom_line(size=0.05) +
  geom_point() +
  labs(x = "error", y = "unfairness") +
  theme_bw(base_size=13)
ggsave("img/ndf_adult_fine.pdf", dpi=300, width=3, height=4)










