###################
# body.R
# 
# Create the body for the ui. 
# If you had multiple tabs, you could split those into their own
# components as well.
###################
body <- dashboardBody(
  
  tags$head(tags$style(HTML('
                            /* logo */
                            .skin-blue .main-header .logo {
                            background-color: #426778;/*sayfa içi*/
                            }
                            
                            /* logo when hovered */
                            .skin-blue .main-header .logo:hover {
                            background-color: #426778;
                            
                            
                            }
                            
                            
                            
                            /* navbar (rest of the header) */
                            .skin-blue .main-header .navbar {
                            background-color: #426778;/*en üst*/
                            
                            }
                            
                            /* main sidebar */
                            .skin-blue .main-sidebar {
                            background-color: #426778;
                            }
                            
                            
                            /* active selected tab in the sidebarmenu */
                            .skin-blue .main-sidebar .sidebar .sidebar-menu .active a{
                            background-color: #749aab;
                            }
                            
                            /* other links in the sidebarmenu */
                            .skin-blue .main-sidebar .sidebar .sidebar-menu a{
                            background-color: #426778;
                            color: #ffffff;
                            }
                            
                            /* other links in the sidebarmenu when hovered */
                            .skin-blue .main-sidebar .sidebar .sidebar-menu a:hover{
                            background-color: #e8b3b5;
                            }
                            /* toggle button when hovered  */
                            .skin-blue .main-header .navbar .sidebar-toggle:hover{
                            background-color: #e8b3b5;
                            }
                            
                            /* body */
                            .content-wrapper, .right-side {
                            background-color: #c5dae3;
                            
                            }
                            .sidebar-toggle {
                            float: top;
                            }
                            
                            '))),
  tabItems(
    
    ########################
    # First tab content
    ########################
    tabItem(
      tabName = "dashboard",
      fluidRow(
        
        # CONTROLS
        box(
          
          title = "Controls",
          solidHeader = TRUE,
          status = "warning",
          # Choose a column
          selectInput(
            "columnChoice",
            "Choose a column:",
            choices = colnames(df),
            selected = "n"),
          
          sliderInput("slider", "Number of observations:", 1, 100, 50),
          
          
          
          # Create an eventReactive element
          actionButton(
            inputId = "submit",
            label = "Submit column")
          
        ),
        # PLOT THE THTINGS
        box(title = "histPlot", status = "primary", plotOutput("histPlot", height = 250))
      )
    ),
    
    ########################
    # Second tab content
    ########################
    tabItem(
      tabName = "widgets",
      h2("Widgets tab content")
    )
  )
  )
