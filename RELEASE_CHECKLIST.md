# Publishing the standalone BYOD repository

The intended repository is `https://github.com/RuurdKuiper/BYOD`, matching the
manuscript, notebook, package metadata, and README. Publish it before submitting
the manuscript's code-availability statement to arXiv. Original software uses
MIT; models, datasets, and upstream dependencies retain their own terms.

## Export and publish

Run from this folder:

```bash
python prepare_release.py
cd dist/BYOD
git init -b main  # only if not already initialized
git add .
git diff --cached --stat
git commit -m "Release BYOD training and evaluation code"
```

Create an empty GitHub repository `RuurdKuiper/BYOD` without an initial README
or license. From the clean checkout:

```bash
git remote add origin git@github.com:RuurdKuiper/BYOD.git
git push -u origin main
```

The export has no parent Git history, downloaded data, checkpoints, caches, or
`.env` file. `.env.example` has a placeholder token. Review staged files before
committing. If the remote was configured during preparation, skip adding it.

## Verify public access

- Open the repository and Colab link while logged out.
- Run `pip install -e '.[test,evaluation]'` and `pytest -q` in the standalone
  checkout. GitHub Actions runs the CPU checks on pushes and pull requests.
- Smoke-test each small-model preset with `--mode quick --max-updates 2` on CUDA.
  A GPU smoke test is still needed; local CPU checks do not prove GPU compatibility.
- Open all demos and the paper's dataset links without privileged access.
- Add the arXiv identifier to `CITATION.cff` after announcement.

`REPRODUCIBILITY.md` records the author-confirmed 0.9/0.1 loss correction and the
scope of the available historical configurations. Do not describe the retained
historical Llama file as complete provenance for every reported checkpoint.
