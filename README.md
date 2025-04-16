# GLGCN  
A multi-omics integration method based on graph-level structure fusion and locus-level feature fusion for cancer subtype classification.

## Overview  
![GCN4_00](https://github.com/user-attachments/assets/11f72ef3-2d56-45de-af8e-40eb66105177)
As shown in figure, in the graph-level structure fusion module, we first construct gene-gene graphs based on multi-omics data, integrate multi-omics information using SNF, and optimize the graph structure through structural entropy-based edge filtering and PPI network-based edge supplementation. In the locus-level feature fusion module, we integrate gene features from different omics layers using a LGCN, preserving the unique information of each omics while capturing complementary characteristics across omics. This process generates gene-level fused representations and predicts cancer subtypes, enabling efficient classification.

## Requirements  
To run GLGCN, ensure the following dependencies are installed:  
- Python 3.9+
- PyTorch 1.10.1+
- NumPy  1.25.1+
- Pandas  2.0.3+
- Snfpy 0.2.2 
- Scikit-learn  1.3.0+
- NetworkX  3.1


## Files  
- `GLGCN_run.py`: Main script to execute the entire GLGCN pipeline for cancer classification.  
- `LCGN_model.py`: Defines the SLGCN model architecture for locus-level feature fusion and classification.  
- `PPI.py`: Handles the integration of Protein-Protein Interaction (PPI) data from the STRING database to supplement network edges.  
- `README.md`: Project documentation.  
- `SE.py`: Implements the structural entropy-based edge filtering for network optimization.  
- `SNF.py`: Performs Similarity Network Fusion to integrate gene similarity graphs from mult-omics.  
- `__init__.py`: Initializes the Python package.  
- `biomarker.py`: Identifies and processes biomarkers relevant to cancer subtype classification.  
- `layer.py`: Defines the graph convolutional layers used in the SLGCN model.  
- `preprocess.py`: Preprocesses multi-omics data (e.g., Met, SCNV, RNA) to select sample intersections.  
- `utils.py`: Contains utility functions for loading data.  

## Usage  
1. **Prepare Data**: Ensure your multi-omics data is formatted as required (refer to `preprocess.py` for details).
2. **Graph-level structure fusion**: Run SNF.py to perform Similarity Network Fusion and obtain the fused gene similarity graph. Use the following command:
   ```bash
   python SNF.py -rd data/BRCA/DEG_results/Met.txt data/BRCA/DEG_results/SCNV.txt data/BRCA/DEG_results/Seq_RNA.txt -fd data/BRCA/Met.csv data/BRCA/SCNV.csv data/BRCA/Seq_RNA.csv --metric cosine
   ```
   - `-rd`: Path of significant gene directory screened by differential analysis.  
   - `-fd`: Path to the multi-omics data directory.
   
4. **Run the Pipeline**: Execute the main script to perform the entire workflow:  
   ```bash
   python GLGCN_run.py -rd data/BRCA/DEG_results/Met.txt data/BRCA/DEG_results/SCNV.txt data/BRCA/DEG_results/Seq_RNA.txt -fd data/BRCA/Met.csv data/BRCA/SCNV.csv data/BRCA/Seq_RNA.csv -fad results/BRCA-SNF.csv -ld data/BRCA/label.csv
   ```
   - `-fad`: Path of the fused graph structure directory.
   - `-ld`: Path of the sample label directory.

For detailed implementation of each module, refer to the corresponding Python scripts (e.g., `SNF.py` for network fusion, `LCGN_model.py` for the SLGCN model).


