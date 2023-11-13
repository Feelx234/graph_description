from pathlib import Path

def get_dataset_folder():

    path = Path(__file__).parent.absolute()

    if str(path.name) == "scripts":
        path = path.parent
    if str(path.name) == "graph_description":
        path = path.parent
    if str(path.name) == "src":
        path = path.parent
    if str(path.name)!="datasets":
        path = path/"datasets"

    assert path.is_dir()
    return path