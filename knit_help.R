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



# === Tree-based ===#
quarto::quarto_render(here::here("LectureNotes/P01-random-forest.qmd"))
quarto::quarto_render(here::here("LectureNotes/P02-boosted-regression-forest.qmd"))

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
#' # Extensions
#/*=================================================*/
quarto::quarto_render(here::here("LectureNotes/E01-spatial-cv.qmd"))

#/*=================================================*/
#' # Appendix
#/*=================================================*/
# === MC ===#
quarto::quarto_render(here::here("LectureNotes/A01-mc-simulation.qmd"))



