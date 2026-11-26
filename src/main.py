
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"


def run_module(filename, description):
    script_path = SRC_DIR / filename

    print("\n" + "=" * 65)
    print(description)
    print("=" * 65)

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_DIR,
        check=True
    )

    print(f"\n{description} completed successfully.")


def main():
    print("=" * 65)
    print("HEALTHCARE INFORMATION RETRIEVAL SYSTEM")
    print("=" * 65)

    try:
        run_module(
            "positional_index.py",
            "PART A - POSITIONAL INDEX"
        )

        run_module(
            "test_positional_index.py",
            "PART A - POSITIONAL INDEX VALIDATION"
        )

        run_module(
            "retrieval.py",
            "PART B - TF, DF, IDF, TF-IDF AND NORMALIZATION"
        )

        run_module(
            "retrieval_test.py",
            "PART B - RETRIEVAL VALIDATION"
        )

        run_module(
            "phrase_queries.py",
            "PART D - TERM, PHRASE AND BOOLEAN QUERIES"
        )

        print("\n" + "=" * 65)
        print("PROJECT EXECUTION COMPLETED SUCCESSFULLY")
        print("=" * 65)

    except subprocess.CalledProcessError as error:
        print("\nPROJECT EXECUTION FAILED.")
        print("Failed script:", error.cmd)
        sys.exit(1)


if __name__ == "__main__":
    main()
