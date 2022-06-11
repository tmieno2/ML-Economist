library(knitr)
library(rmarkdown)
library(here)
library(quarto)
library(xaringanBuilder)

# /*===========================================================
#' # Render lecture notes
# /*===========================================================
quarto_render(here("LectureNotes/bias-variance-tradeoff.qmd"))
quarto_render(here("LectureNotes/cross-validation.qmd"))

