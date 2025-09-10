# SimUFD

This repository contains the implementation code for the paper ["An unsupervised approach to early fault detection and performance degradation assessment in bearings"](https://doi.org/10.1016/j.aei.2025.103620) published in *Advanced Engineering Informatics*.


## Project Structure
```text
├── Main.py                     # Main entry point for training/testing
├── model/
│   ├── VAE.py                  # VAE model and trainer
│   └── model_base.py           # Base trainer class
├── src/
│   ├── utils.py                # Utility functions and dataset classes
│   ├── models.py               # Model selection utilities 
│   └── dataset_configs.py      # Dataset configuration and data loader
├── dataset/                    # Raw data files
│   ├── IMS/
│   └── XJ/
├── trained_models_XJ_2_2/      # Saved model checkpoints
└── README.md                   # Project documentation
```

## Usage
1. Prepare your dataset in the `dataset/` folder as described above.  
   The data should be stored in MATLAB `.mat` files, where each file contains a matrix (e.g., `IMS_2_1`) with shape `[num_samples, num_features]`.  Here, `num_features` refers to the number of channels in the data. 
2. Run the main script for training/testing:
   ```bash
   python Main.py -data_type XJ_2_2 -model SimUFD 
   ```

## Arguments
Key command-line arguments:
- `-data_type`: Dataset type (e.g., IMS_2_1, XJ_2_2)
- `-model`: Model architecture (default: SimUFD)

See `Main.py` for all available arguments.

## Results
- Training and validation losses are saved as CSV and PNG files in the `trained_models_<data_type>/` folder.
- Health Index (HI) results and plots are also saved for further analysis.

## Citation
If you find this code useful for your research, please reference the following paper:
```
@article{wang2025unsupervised,
  title={An unsupervised approach to early fault detection and performance degradation assessment in bearings},
  author={Wang, Shun and Vidal, Yolanda and Pozo, Francesc},
  journal={Advanced Engineering Informatics},
  volume={68},
  pages={103620},
  year={2025},
  publisher={Elsevier}
}
```

## License
This project is licensed under the [MIT License](LICENSE).

