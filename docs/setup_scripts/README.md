# Notes on Setup


## Other info related to administering / editing documentation

### Editing the Readme for display in Jupyter book. 

To automate citation formatting for the README document.

> `pandoc -t markdown_strict -citeproc README-draft.md -o README.md --bibliography bib/bibliography.bib`


### Jupyter Book and Binder

Launch the main notebook in "interactive mode" using Binder:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dankovacek/run_of_river_intro.git/main)

Individual notebook files are saved under [content/notebooks/](https://github.com/dankovacek/bcub/tree/main/content/).

### Notes on Compiling and Updating the Book 

Info for [building books and hosting on Github Pages](https://jupyterbook.org/publish/gh-pages.html)

After updating any content, rebuilt the repo:

`jupyter-book build content/`

Then, update the github pages site. Use the gh-pages branch update tool:

`ghp-import -n -p -f content/_build/html`

[Visit the site](https://dankovacek.github.io/Engineering_Hydrology_Notebooks/) at Github sites

`https://dankovacek.github.io/Engineering_Hydrology_Notebooks/`

### Large file storage

See [this answer from github](https://stackoverflow.com/a/48734334/4885029).  

```
$ curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
$ sudo apt-get install git-lfs
$ git-lfs install
```




