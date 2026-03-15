Yo bitches (skrevet av sivert)

## Important!!

Jupyter files stores metadata such as execution_count, outputs and others which we dont want to push to github. To filter the metadata out use the library nbstripout,

so before commiting anything jupyter related, install:

```
pip install nbstripout
```

In the folder with the .git file (/RegModel) run:

```
nbstripout --install
```
