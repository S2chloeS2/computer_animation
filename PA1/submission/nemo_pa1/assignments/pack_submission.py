import zipfile
from pathlib import Path
from gitignore_parser import parse_gitignore
from rich import print as rprint


# pack all necessary files for submission into a zip file
# while ignoring files specified in .gitignore
# zip file will be named as FOLDER_NAME_PA_ID.zip

PA_ID = "pa1"
FOLDER_NAME = "nemo"


def create_zip_ignoring_gitignore(zip_filename, source_dir):
    """
    Creates a zip archive of the source directory, ignoring files specified in .gitignore.
    """
    # Get the list of all untracked and tracked files in the repo
    rprint(f"Checking [green]{source_dir}[/green] for submission ...")
    try:
        gitignore = parse_gitignore(source_dir / ".gitignore")

        all_files = source_dir.rglob("*")
        # Use the 'ignored' method to get the list of files that ARE ignored
        # The paths passed to repo.ignored need to be absolute paths
        abs_all_files = [f.absolute() for f in all_files if f.is_file()]
        # Filter out the ignored files from the total list
        files_to_zip = [f for f in abs_all_files if not gitignore(f)]

    except FileNotFoundError:
        rprint(f"[yellow][bold]Warning:[/bold][/yellow] No .gitignore file found at [red]{source_dir}[/red]. Zipping all files without .gitignore rules.")
        all_files = source_dir.rglob("*")
        files_to_zip = [f for f in all_files if f.is_file()]

    # Create the zip file
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in files_to_zip:
            # Add file to zip, preserving relative path structure
            relative_path = file_path.relative_to(source_dir)
            p0 = relative_path.parts[0]
            if p0 in [".git", ".venv", ".cache"]:
                continue
            rprint(f"Adding [green]{relative_path}[/green] ...")
            if relative_path != ".":  # Avoid adding the current directory itself
                zipf.write(file_path, arcname=relative_path)

    rprint(f"Successfully created zip archive: {zip_filename}")
    return zip_filename


def iter_parent_dirs(start_path="."):
    """Yield all parent directories from start_path up to the filesystem root."""
    path = Path(start_path).resolve()
    yield from path.parents


if __name__ == "__main__":
    found_src = False
    for parent in iter_parent_dirs(__file__):
        if parent.name.startswith(FOLDER_NAME):
            found_src = True
            ret_zip = create_zip_ignoring_gitignore(f"{FOLDER_NAME}_{PA_ID}.zip", parent)
            rprint(f"\nYou can now submit the zip file to Courseworks: [green][bold]{ret_zip}[/bold][/green]")
            break
    if not found_src:
        raise ValueError(f"Source directory {FOLDER_NAME} not found")
