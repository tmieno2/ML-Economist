library(knitr)
library(rmarkdown)
library(here)
library(quarto)
library(xaringanBuilder)

# /*===========================================================
#' # Render lecture notes
# /*===========================================================

# === bias-variance trade-off ===#
quarto::quarto_render(here("LectureNotes/bias-variance-tradeoff.qmd"))

# === cross-validation ===#
quarto::quarto_render(here("LectureNotes/cross-validation.qmd"))

# === bootstrap ===#
quarto::quarto_render(here("LectureNotes/bootstrap.qmd"))# === bootstrap ===#

#=== LASSO ===#
quarto::quarto_render(here("LectureNotes/lasso.qmd"))

#=== Tree-based ===#
quarto::quarto_render(here("LectureNotes/tree-based-ml.qmd"))