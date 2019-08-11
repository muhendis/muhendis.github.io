###################
# global.R
# 
# Anything you want shared between your ui and server, define here.
###################
library(shinydashboard)


set.seed(122)
df <- data.frame(
  n = rnorm(500),
  m = rnorm(100))
