library(knitr)
library(rmarkdown)
library(here)
library(quarto)
library(xaringanBuilder)

# /*===========================================================
#' # Render lecture notes
# /*===========================================================

# === prediction v.s. causal ===#
quarto::quarto_render(here::here("LectureNotes/preface.qmd"))

# === non-linear ===#
quarto::quarto_render(here::here("LectureNotes/B01-nonlinear.qmd"))

# === bias-variance trade-off ===#
quarto::quarto_render(here::here("LectureNotes/B02-bias-variance-tradeoff.qmd"))

# === LASSO ===#
quarto::quarto_render(here::here("LectureNotes/B04-regularization.qmd"))

# === cross-validation ===#
quarto::quarto_render(here::here("LectureNotes/B03-cross-validation.qmd"))

# === bootstrap ===#
quarto::quarto_render(here::here("LectureNotes/B05-bootstrap.qmd")) # === bootstrap ===#

#/*=================================================*/
#' # Prediction (forest-based)
#/*=================================================*/
# === Tree-based ===#
quarto::quarto_render(here::here("LectureNotes/P01-random-forest.qmd"))
quarto::quarto_render(here::here("LectureNotes/P02-boosted-regression-forest.qmd"))
quarto::quarto_render(here::here("LectureNotes/P03-xgb.qmd"))

#/*=================================================*/
#' # Causal ML
#/*=================================================*/
# === double-ml ===#
quarto::quarto_render(here::here("LectureNotes/C01-dml.qmd"))

# === het te ===#
quarto::quarto_render(here::here("LectureNotes/C02-het-te.qmd"))

# === grf ===#
quarto::quarto_render(here::here("LectureNotes/C04-grf.qmd"))

# === cf ===#
quarto::quarto_render(here::here("LectureNotes/C03-cf-orf.qmd"))

# === model selection ===#
quarto::quarto_render(here::here("LectureNotes/C05-model-selection.qmd"))

# === scikitlearn ===#
quarto::quarto_render(here::here("LectureNotes/scikitlearn.qmd"))

# === linear-nonpar ===#
quarto::quarto_render(here::here("LectureNotes/linear-nonparametric-efficiency.qmd"))

#/*=================================================*/
#' # Programming
#/*=================================================*/
#=== mlr3 ===#
quarto::quarto_render(here::here("LectureNotes/PROG-01-mlr3.qmd"))

#=== reticulate ===#
quarto::quarto_render(here::here("LectureNotes/PROG-02-reticulate.qmd"))

#=== model selection ===#
quarto::quarto_render(here::here("LectureNotes/PROG-03-model-selection-prediction.qmd"))

#/*=================================================*/
#' # Extensions
#/*=================================================*/
quarto::quarto_render(here::here("LectureNotes/E01-spatial-cv.qmd"))

#/*=================================================*/
#' # Appendix
#/*=================================================*/
# === MC ===#
quarto::quarto_render(here::here("LectureNotes/A01-mc-simulation.qmd"))



