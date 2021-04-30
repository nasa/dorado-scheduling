# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

from pathlib import Path
import sys

from packaging.version import Version
import pep517.meta

parent_path = str(Path(__file__).parents[1])
sys.path.insert(0, parent_path)


# -- Project information -----------------------------------------------------

metadata = pep517.meta.load(parent_path).metadata

project = metadata['name']
author = metadata['author']
release = metadata['version']
version = Version(metadata['version']).public
copyright = '''2020, United States Government as represented by the \
Administrator of the National Aeronautics and Space Administration'''


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autosummary',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx_gallery.gen_gallery',
    'sphinxarg.ext'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

modindex_common_prefix = ['dorado.scheduling.']


# -- Options for autodoc -----------------------------------------------------

autodoc_default_options = {
    'members': True,
    'show-inheritance': True
}
autodoc_member_order = 'bysource'
autosummary_generate = True


# -- Options for extlinks extension ------------------------------------------

extlinks = {
    'arxiv': ('https://arxiv.org/abs/%s', 'arXiv:'),
    'doi': ('https://doi.org/%s', 'doi:')
}


# -- Options for sphinx_gallery extension ------------------------------------

sphinx_gallery_conf = {
    'examples_dirs': ['../examples'],
    'gallery_dirs': ['examples'],
}


# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'astroplan': ('https://astroplan.readthedocs.io/en/latest/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
}


# -- Options for plot_directive ----------------------------------------------

plot_include_source = True
plot_formats = [('svg', 300), ('pdf', 300)]
plot_html_show_formats = False


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    "github_url": f"https://github.com/nasa/{project}",
    "icon_links": [
        {
            "name": "PyPI",
            "url": f"https://pypi.org/project/{project}",
            "icon": "fas fa-box",
        }
    ]
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_use_modindex = True
