## Heterogeneous graph representation learning on protein-protein-metabolite interaction networks

This repository provides an implementation of [MAGNN](https://github.com/cynricfu/MAGNN) applied to the metabolite concentration change sign prediction task as described in the thesis manuscript by Nathan van der Lei, MSc Computational Science, UvA/VU, Amsterdam. This repository was developed in collaboration with the [Fraenkel Lab](http://fraenkel.mit.edu/), Department of Biological Engineering, MIT.

The thesis manuscript itself will become available at [Theses University of Amsterdam](https://scripties.uba.uva.nl)

### Data download
1. Download "Data.zip" folder from [Dropbox](https://www.dropbox.com/s/iikwrukbtwg8l2n/Data.zip?dl=0)
2. Unzip "Data.zip" to the same parent directory as this repository
3. Your file structure should look as intended:
  - 📁 "YOUR parent directory"
    - 📄 Data.zip
    - 📁 Data
    - 📁 PPMI_MAGNN
    - ...  

### Usage
1. Download data (as described previously)
2. Use `preprocess_PPMI.ipynb` to prepare data for MAGNN and create / load a `Dataloader` object
3. Use `baselines.ipynb` to train and test vanilla ML methods
4. Use `network_analysis.ipynb` to investigate the PPMI network
5. Create `checkpoint/` folder in repository for model saving
6. Run the command `python run_PPMI.py` from terminal to train MAGNN
7. Use `inspect_model.ipynb` to investigate a trained model, access embeddings and visualize nodes
8. Use `compare_results.ipynb` to visualize experimental results

Run `python run_PPMI.py --help` for more information about the available options of the PPMI MAGNN implementation. 

### Project overview
The implementation uses 4 data components:
1. **PPMI network**: under active development at the [Fraenkel Lab](https://github.com/fraenkel-lab)
2. **Metabolite attributes**: with data from [HMDB 4.0 download](https://hmdb.ca/downloads)
3. **Protein attributes**: with data from [Protein Atlas download](https://www.proteinatlas.org/search)
4. **Metabolite class labels**: with data from San-Millán et al. (2020), available at [10.3389/fphys.2020.00578](https://doi.org/10.3389/fphys.2020.00578)

![](images/methods_overview.png)

### Experimental setup

|   Method      | Uses PPMI network | Uses metabolite attributes | Uses protein attributes | Uses node class labels |
|--------------:|:-----------------:|:--------------------------:|:--------------------:|:----------------------:|
| **Baselines** |                   |              x             |                      |            x           |
|  **MAGNN 1**  |         x         |                            |                      |            x           |
|  **MAGNN 2**  |         x         |              x             |                      |            x           |
|  **MAGNN 3**  |         x         |              x             |           x          |            x           |

### QA

**Q: Where can I find relevant code?** <br>
A: Some relevant code snippets are mentioned here:
- The `DataLoader` class in `PPMI_dataloader.py` can be used to perform the data preprocessing.
- The `ExerciseMetabolomicsDataLoader` can be used to obtain mean log2fold change values per metabolite from a publication
- The `load_PPMI_data` function in `utils/data.py` is build to load data into MAGNN specific format.
- The `get_metapath_neighbor_pairs_PPMI_MP` function in `preprocess_PPMI.ipynb` can be used to obtain metapath instances by brute force search. The search is parallelized using `multiprocessing` for computational optimization.

**Q: How does the data preprocessing work?** <br>
A: The preprocessing happens at initalization of a `DataLoader` object in `PPMI_dataloader.py`. Depending on some preprocessing settings the correct data is retrieved, combined and stored in the `DataLoader` class. 

![](images/preprocessing_workflow.png)

**Q: Where can I find the documentation?** <br>
A: The codebase is designed to be self contained and all relevant documentation can be found in the code itself. It is assumed users are fluent in Python and are able to work with jupyter notebooks and terminal scripts by themselves.

**Q: Is there a Docker image available?** <br>
A: currently there is no Docker image available for this project.

**Q: How can I make predictions for a new metabolomics publication?** <br>
A: The publication should contain supplementary data on metabolite abundances pre and post intervention. The data files should be uploaded into the correct data folder in the correct format.

**Q: How to cite this work?**
A: if you find this work useful, please consider citing:

    @MastersThesis{vanderlei_2021_heterogeneous,
        title={Heterogeneous graph representation learning on protein-protein-metabolite interaction networks},
        author={van der Lei, Nathan M},
        school={University of Amsterdam}
        year={2021}}  

### Resources

- Sportomics publication used to obtain metabolite class labels:
> Metabolomics of Endurance Capacity in World Tour Professional Cyclists <br>
> San-Millán, I., Stefanoni, D., Martinez, J. L., Hansen, K. C., D’Alessandro, A., & Nemkov, T. <br>
> Frontiers in Physiology, 2020, 11: 578 <br>
> DOI: [10.3389/fphys.2020.00578](https://doi.org/10.3389/fphys.2020.00578)

- Code implemention builds on top of MAGNN:
> MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding.<br>
> Xinyu Fu, Jiani Zhang, Ziqiao Meng, Irwin King.<br>
> The Web Conference, 2020. <br>
> EPrint: [arXiv:2002.01680](https://arxiv.org/abs/2002.01680)

<!-- - [Markdown-cheat-sheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)
- [Emoji-cheat-sheet](https://github.com/ikatyang/emoji-cheat-sheet) -->

### Dependencies

Recent versions of the following packages for Python 3 are required:
* PyTorch 1.2.0
* DGL 0.3.1
* NetworkX 2.3
* scikit-learn 0.21.3
* NumPy 1.17.2
* SciPy 1.3.1

### Other benchmark datasets

The preprocessed datasets used in MAGNN benchmarking are available at:
* IMDb - [Dropbox](https://www.dropbox.com/s/g0btk9ctr1es39x/IMDB_processed.zip?dl=0)
* DBLP - [Dropbox](https://www.dropbox.com/s/yh4grpeks87ugr2/DBLP_processed.zip?dl=0)
* Last.fm - [Dropbox](https://www.dropbox.com/s/jvlbs09pz6zwcka/LastFM_processed.zip?dl=0)

<!-- The GloVe word vectors are obtained from [GloVe](https://nlp.stanford.edu/projects/glove/). Here is [the direct link](http://nlp.stanford.edu/data/glove.6B.zip) for the version we used in DBLP preprocessing.
 -->
### Usage of benchmark datasets

1. Create `checkpoint/` and `data/preprocessed` directories
2. Extract the zip file downloaded from the section above to `data/preprocessed`
    * E.g., extract the content of `IMDB_processed.zip` to `data/preprocessed/IMDB_processed`
2. Execute one of the following three commands from the project home directory:
    * `python run_IMDB.py`
    * `python run_DBLP.py`
    * `python run_LastFM.py`

For more details, please visit the original [MAGNN](https://github.com/cynricfu/MAGNN) repository. 