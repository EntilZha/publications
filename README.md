# Main Idea

In the interest of transparency and blind accessibility, we release the LaTeX
source of our publications along with the python code to generate the figures
from the paper's data.

# Organization

Each paper has its own directory of the form

> YEAR_VENUE_KEYWORDS

Year is the year of publication, venue is where it was published, and keywords
are some uniquely identifying terms (not always the title, which can sometimes
be more cutesy).

# Requirements

To compile papers, you will need:

- PDFLatex
- bibtex
- Python
- [Poetry](https://python-poetry.org)

# Compiling

To compile the PDF of a paper (e.g., 2020_emnlp_curiosity), there are three steps

1. If the paper subdirectory has a `poetry.lock` file, then it has python dependencies you need to install and you will need to activate the python environment
1. Navigate to paper directory: `cd YEAR_VENUE_KEYWORDS`
1. Create the virtual environment: `poetry install`
1. Activate the virtual environment: `poetry shell`
1. Navigate to repository root: `cd ..`
1. Run: `make 2020_curiosity.paper.pdf`

This generates the PDF, compiles the bibliography, and creates any figures
needed by the file.

If there are errors due to missing data, consult the readme for the particular paper. Its likely that some of the input data (like the dataset to analyze), is too large to reasonably commit to git.

# Screenreaders

To use a screenreader to read the source, open the main file:

> 2020_emnlp_curiosity.tex

And then follow the input commands to find any included files.

Or you can go to the "sections" subdirectory within a paper and read the LaTeX
files in order (prefixed by number to help you read in order).
