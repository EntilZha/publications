.DEFAULT_GOAL := paper
TEX = $(wildcard */sections/*.tex *.tex */*.tex)
GFX = $(wildcard */figures/*.*)
DATA = $(wildcard */data/*.*)
BIB = $(wildcard bib/*.bib)
FIG = $(wildcard */figures.py */figures/*.jinja2)

# We don't want make to delete bibliography files or the figures, so we need this rule
.SECONDARY:
.PRECIOUS: %/auto_fig/res.txt

clean:
	rm -f *.aux *.dvi *.log *.bbl *.pdf *~ *.out *.blg *.nav *.toc *.snm *.fdb_latexmk *.fls *.synctex.gz
	rm -f 2020_emnlp_curiosity/*.aux 2020_emnlp_curiosity/*.dvi */*.log */*.bbl 2020_emnlp_curiosity/*.pdf */*~ */*.out */*.blg */*/*~
	rm -fR */auto_fig/*
	rm *.arxiv.tgz

%.bbl: $(BIB) $(TEX)
	pdflatex -interaction=nonstopmode -halt-on-error $*
	bibtex $*

%/auto_fig:
	mkdir -p $@

%/auto_fig/res.txt: %/auto_fig $(DATA) $(FIG)
	bash scripts/rscript_if_ne.sh $(<:/auto_fig=) > $@

# %.tex needs to be the first dependency or it will cause an error
%.paper.pdf: %.tex %/auto_fig/res.txt %.bbl $(GFX)
	pdflatex -interaction=nonstopmode -halt-on-error $*
	pdflatex -interaction=nonstopmode -halt-on-error $*
	cp $(<:.tex=.pdf) $@
	ruby scripts/style-check.rb $(<:.tex=)/*.tex $(<:.tex=)/sections/*.tex

# These targets should remain in sync (e.g., if you fix one, do the same for the other).  Except for the .tgz target should have all the bib files but the arxiv target should have the bbl file ($<)
%.tgz: %.bbl
	tar cvfz $@ Makefile style/*.sty style/*.bst $(<:.bbl=.tex) bib/*.bib style/*.tex $(<:.bbl=)/figures/* $(<:.bbl=)/auto_fig/* $(<:.bbl=)/commit_auto_fig/* $(<:.bbl=)/sections/*.tex

%.arxiv.tgz: %.bbl
	tar cvfz $@ Makefile $< style/*.sty style/*.bst $(<:.bbl=.tex) style/*.tex $(<:.bbl=)/figures/* $(<:.bbl=)/auto_fig/* $(<:.bbl=)/commit_auto_fig/* $(<:.bbl=)/sections/*.tex

paper: 2020_emnlp_curiosity.paper.pdf
