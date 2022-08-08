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
# === RF ===#
quarto::quarto_render(here::here("LectureNotes/P01-random-forest.qmd"))

#=== GB ===#
quarto::quarto_render(here::here("LectureNotes/P02-boosted-regression-forest.qmd"))

#=== XGB ===#
quarto::quarto_render(here::here("LectureNotes/P03-xgb.qmd"))

#=== local linear forest ===#
quarto::quarto_render(here::here("LectureNotes/P04-local-linear-forest.qmd"))

#/*=================================================*/
#' # Causal ML
#/*=================================================*/
#=== why not this? ===#
quarto::quarto_render(here::here("LectureNotes/C00-why-not-this.qmd"))

# === double-ml ===#
quarto::quarto_render(here::here("LectureNotes/C01-dml.qmd"))

# === XSTR ===#
quarto::quarto_render(here::here("LectureNotes/C02-xstr-learner.qmd"))

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

#=== grf ===#
quarto::quarto_render(here::here("LectureNotes/E02-grf.qmd"))

#/*=================================================*/
#' # Appendix
#/*=================================================*/
# === MC ===#
quarto::quarto_render(here::here("LectureNotes/A01-mc-simulation.qmd"))

# === MM ===#
quarto::quarto_render(here::here("LectureNotes/A02-method-of-moment.qmd"))



