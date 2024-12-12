"""Generate JSON schemas for tasks and write them to the Fractal manifest."""

from fractal_tasks_core.dev.create_manifest import create_manifest

if __name__ == "__main__":
    PACKAGE = "scmultiplex"
    AUTHORS = "Nicole Repina, Enrico Tagliavini, Tim-Oliver Buchholz, Joel Luethi"
    docs_link = "https://github.com/fmi-basel/gliberal-scMultipleX"
    if docs_link:
        create_manifest(package=PACKAGE, authors=AUTHORS, docs_link=docs_link)
    else:
        create_manifest(package=PACKAGE, authors=AUTHORS)
