## Installation

The tool is a single script. The only setup needed is a Python environment with a few packages and an editor to run it from.

- Download and install [Miniconda](https://www.anaconda.com/download/success) for your operating system.

- On Linux, make sure a suitable graphics driver is installed and hardware acceleration is enabled, since the interface relies on Qt windows.

- Open a conda-enabled terminal:
   - **Windows**: search for "Anaconda Prompt" in the Start menu.
   - **Linux and macOS**: open a terminal. If `conda` is not on your path, activate it first:
     ```bash
     source ~/miniconda3/bin/activate    # Anaconda: source ~/anaconda3/bin/activate
     ```

- Create and activate the environment:
   ```bash
   conda tos accept
   conda create -n artifactscorer -c conda-forge python=3.12 numpy pandas scipy matplotlib joblib pyedflib pyqt spyder -y
   conda activate artifactscorer
   pip install lspopt
   spyder
   ```
- Set the plotting backend to Qt, which the interface requires: **Tools → Preferences → IPython console → Graphics → Backend → Qt5**, then restart the console. Alternatively, run `%matplotlib qt` in the console before each session. Without this, the script stops with `RuntimeError: Qt backend required`.

- Clone this repository, or just [download the main script](https://github.com/Niloy333/EEG-artifact-scorer/blob/base/eeg_2ch_artifact_scorer.py), open it in Spyder, and run it.
