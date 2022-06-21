library(knitr)
library(rmarkdown)
library(here)
library(quarto)
library(xaringanBuilder)

# /*===========================================================
#' # Render lecture notes
# /*===========================================================

# === prediction v.s. causal ===#
quarto::quarto_render(here::here("LectureNotes/L00-prediction-vs-causal-inference.qmd"))

# === bias-variance trade-off ===#
quarto::quarto_render(here::here("LectureNotes/bias-variance-tradeoff.qmd"))

# === gam ===#
quarto::quarto_render(here::here("LectureNotes/gam.qmd"))

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

# === scikitlearn ===#
quarto::quarto_render(here::here("LectureNotes/scikitlearn.qmd"))

# === linear-nonpar ===#
quarto::quarto_render(here::here("LectureNotes/linear-nonparametric-efficiency.qmd"))


#/*=================================================*/
#' # Appendix
#/*=================================================*/
# === MC ===#
quarto::quarto_render(here::here("LectureNotes/A01-mc-simulation.qmd"))



