"""Generate JSON schemas for tasks and write them to the Fractal manifest."""

from fractal_tasks_core.dev.create_manifest import create_manifest

if __name__ == "__main__":
    PACKAGE = "scmultiplex"
    AUTHORS = "Nicole Repina, Enrico Tagliavini, Tim-Oliver Buchholz, Joel Lüthi"
    create_manifest(package=PACKAGE, authors=AUTHORS)
