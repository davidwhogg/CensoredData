LATEX       = pdflatex
RM          = rm -rf
RM_TMP      = ${RM} $(foreach suff, ${TMP_SUFFS}, ${NAME}.${suff})

TMP_SUFFS   = pdf aux bbl blg log dvi ps eps
SUFF        = pdf

CHECK_RERUN = grep Rerun $*.log

.SUFFIXES: .tex .dvi .ps .pdf

