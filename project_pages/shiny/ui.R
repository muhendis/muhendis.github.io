###################
# ui.R
# 
# Initializes the ui. 
# Used to load in your header, sidebar, and body components.
###################
source('./components/header.R')
source('./components/sidebar.R')
source('./components/body.R')


ui <- dashboardPage(
  body = body,                  
  header = header,
  sidebar =  sidebar
  )
