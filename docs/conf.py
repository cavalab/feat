# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../'))
# sys.path.insert(0, os.path.abspath('../../'))
# sys.path.insert(0, os.path.abspath('../python/'))


# -- Project information -----------------------------------------------------

project = 'FEAT'
copyright = '2016, William La Cava, Tilak Raj Singh, University of Pennsylvania'
author = 'William La Cava, Tilak Raj Singh, University of Pennsylvania'

with open('../feat/versionstr.py','r') as f:
    versionstr = f.read().strip()
# The short X.Y version
version = versionstr
# The full version, including alpha/beta/rc tags
release = versionstr


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.linkcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'sphinx_copybutton',

    'sphinx_math_dollar',
    # 'recommonmark',
    'nbsphinx',
    'lxml_html_clean',
    # "sphinx.ext.viewcode",
    # External stuff
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:

# source_suffix = ['.md']

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_material'
html_title = 'FEAT'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {
#         'color_primary': 'white',
#         'color_accent': 'Deep Purple',
# 	# Visible levels of the global TOC; -1 means unlimited
# 	'globaltoc_depth': 3,
# 	# If False, expand all TOC entries
# 	'globaltoc_collapse': False,
# 	# If True, show hidden TOC entries
# 	'globaltoc_includehidden': False,
#         }
# material theme options (see theme.conf for more information)
html_theme_options = {
    "base_url": "https://cavalab.github.io/feat",
    "repo_url": "https://github.com/cavalab/feat/",
    "repo_name": "FEAT",
    "html_minify": False,
    "html_prettify": True,
    "css_minify": True,
    # "logo_icon": "&#120601",
    "logo_icon": "&#120593",
    "repo_type": "github",
    "globaltoc_depth": 1,
    "color_primary": "deep purple",
    "color_accent": "white",
    # 'color_primary': 'white',
    # 'color_accent': 'Deep Purple',
    "touch_icon": "icon.jpeg",
    "theme_color": "#2196f3",
    "master_doc": False,
    # If False, expand all TOC entries
    # 'globaltoc_collapse': True,
    # "nav_links": [
    #     {"href": "index", "internal": True, "title": "Material"},
    #     {
    #         "href": "https://squidfunk.github.io/mkdocs-material/",
    #         "internal": False,
    #         "title": "Material for MkDocs",
    #     },
    # ],
    "heroes": {
        "index": "A tool for learning intelligible models",
        # "customization": "Configuration options to personalize your site.",
    },
    # "version_dropdown": True,
    # "version_json": "_static/versions.json",
    # "version_info": {
    #     "Release": "https://bashtage.github.io/sphinx-material/",
    #     "Development": "https://bashtage.github.io/sphinx-material/devel/",
    #     "Release (rel)": "/sphinx-material/",
    #     "Development (rel)": "/sphinx-material/devel/",
    # },
    "table_classes": ["plain"],
    }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
html_sidebars = {
            "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
            }


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'Featdoc'


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'Feat.tex', 'Feat Documentation',
     'William La Cava, Tilak Raj Singh, University of Pennsylvania', 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'feat', 'Feat Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'FEAT', 'Feat Documentation',
     author, 'FEAT', 'Feature Engineering Automation Tool',
     'Miscellaneous'),
]


# -- Extension configuration -------------------------------------------------
# build Doxygen documentation
import subprocess
subprocess.call('doxygen', shell=True)
# path to Doxygen documentation
html_extra_path = ['doxygen_site'] 

# autodoc_mock_imports = ['feat']
# Render these files as indicated
# source_parsers = {
#                   '.md': 'recommonmark.parser.CommonMarkParser'
#                  }
# Napolean settings
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False

#linkcode resolution
def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    return "https://github.com/cavalab/feat/blob/master/feat/%s.py" % filename

mathjax_config = {
    'tex2jax': {
        'inlineMath': [ ["\\(","\\)"] ],
        'displayMath': [["\\[","\\]"] ],
    },
}

mathjax3_config = {
  "tex": {
    "inlineMath": [['\\(', '\\)']],
    "displayMath": [["\\[", "\\]"]],
  }
}