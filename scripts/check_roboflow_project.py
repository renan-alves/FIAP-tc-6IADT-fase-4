from roboflow import Roboflow

API_KEY = "BOr7518ZsqVWxnoVTNub"
WORKSPACE_NAME = "tcfase4"
PROJECT_NAME = "surgical-instruments-scn9b"

rf = Roboflow(api_key=API_KEY)
workspace = rf.workspace(WORKSPACE_NAME)
project = workspace.project(PROJECT_NAME)

try:
    versions = project.versions()
    if versions:
        for v in versions:
            print(f"Version {v.version}")
    else:
        print("No versions found - project is empty")
except Exception as e:
    print(f"Error: {e}")
