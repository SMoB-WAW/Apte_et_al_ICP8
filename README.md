# C-Trap Force Data & Kymograph Position Analysis

Tools for analysing LUMICKS C-Trap force-extension data and kymographs (`.h5`): finding where green-labelled
proteins bind relative to a red reference marker on DNA, together with the force
applied during each time window.

**Author:** Artur Kaczmarczyk, Single-Molecule Biophysics Facility, IBB PAS Warsaw



script was used to analyze the data in https://doi.org/10.5281/zenodo.22134559


**Girish R. Apte, Shuvankar Patra, Mariusz Czarnocki-Cieciura, Artur P. Kaczmarczyk, Aqsa Jabeen, Justyna Jackiewicz, Jarosław Poznański, Marcin Nowotny, Małgorzata Figiel**
**Structural insights into DNA annealing and recombination by herpesviral DNA-binding proteins ICP8 and BALF2**

## Organization

```
src/kymo_tools.py                     # function library
notebooks/ICP8_CTrap_analysis.ipynb   # example analysis workflow
data/                                 # input .h5 files
results/                              # exported CSVs and figures
```

## Requirements

Python 3 with:

 lumicks.pylake 
 numpy 
 scipy 
 matplotlib 
 pandas 
 peakutils


## Workflow

1. **Load** an `.h5` file and pick a kymograph: `load_h5`, `list_kymos`, `get_kymo`.
2. **Inspect** it with `plot_kymo`, an optional y-flip.
3. **Crop and profile.** Choose a position range (`y_range`) and a time range (`x_range`), then sum the R/G/B intensities over time with `extract_rgb_cumulative_intensity`.
4. **Detect peaks** with `analyze_window`. It finds the red reference peak and the green peaks (via `peakutils`) and reports each green peak's distance from the red one in px and in µm. It also plots the median trap force (pN) over the window.
5. **Plot and export:**
   - `plot_window_analysis` plots the intensity profile with 0 at the red marker.
   - `export_rgb_intensity_csv` and `export_rgb_intensity_plot` save the raw intensity profiles.
   - `export_peak_analysis_csv` writes one row per green peak.
6. **Compare kymographs.** Load every `results/*_peak_analysis.csv`, combine them, and plot a histogram or a strip plot (`plot_peak_comparison`).

s. Load every results/*_peak_analysis.csv, combine them, and plot a histogram or a strip plot (plot_peak_comparison).

