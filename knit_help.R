library(knitr)
library(rmarkdown)
library(here)
library(quarto)
library(xaringanBuilder)

# /*===========================================================
#' # Render lecture notes
# /*===========================================================

# === bias-variance trade-off ===#
quarto::quarto_render(here::here("LectureNotes/bias-variance-tradeoff.qmd"))

# === cross-validation ===#
quarto::quarto_render(here::here("LectureNotes/cross-validation.qmd"))

# === bootstrap ===#
quarto::quarto_render(here::here("LectureNotes/bootstrap.qmd")) # === bootstrap ===#

# === LASSO ===#
quarto::quarto_render(here::here("LectureNotes/lasso.qmd"))

# === Tree-based ===#
quarto::quarto_render(here::here("LectureNotes/tree-based-ml.qmd"))

# === causal-ml ===#
quarto::quarto_render(here::here("LectureNotes/causal-ml.qmd"))

# === grf ===#
quarto::quarto_render(here::here("LectureNotes/grf.qmd"))

# === cf ===#
quarto::quarto_render(here::here("LectureNotes/cf.qmd"))




