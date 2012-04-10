include ../Makefile.inc

NAME    = gm
DOC_OUT = ${NAME}.${SUFF}

default: ${DOC_OUT}

gm.pdf: gm.tex
	latex gm.tex
	dvipdf gm.dvi

clean:
	${RM_TMP}

