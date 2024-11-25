# Fused Magnesium Smelting Process Benchmark

This is a cross-modal benchmark for the fused magnesium smelting process. The benchmark contains a total of 3 hours of synchronously acquired videos and three-phase alternating current data from different production batches. 

## News!

We are in the process of preparing the datasets, which are currently not very convenient for research usage. If you would like to access the dataset in advance, please feel free to contact us: wugc\at mail\dot neu\dot edu\dot cn.

### Dataset Description

This dataset includes three sets of data stored in `.mat` format, comprising $2.7 \times 10^5$ samples with pixel-level annotations. Each file contains the following components:

- **`video`**: A 4D tensor of shape `(height, width, RGB channel, N)` representing the 3D video modality. Here, `N` denotes the number of frames.
  
- **`current`**: A 2D array of shape `(phase channel, N)` representing the 1D three-phase alternating current modality.

- **`label`**: A 3D tensor of shape `(height, width, N)` representing pixel-level normal (`0`) and abnormal (`1`) masks. You can convert these masks to class-level labels using the formula:
  ```matlab
  class_label = single(sum(label, [1, 2]) > 0.5);
  ```

- **`train_test_index`**: A 1D array of shape `(1, N)` indicating the train-test split. A value of `0` represents a training example, while `1` indicates a test example.


## BibTex Citation

If you find this benchmark useful, please cite our paper☺️.
```
@article{wu2024crossmodal,
  title={Cross-Modal Learning for Anomaly Detection in Complex Industrial Process: Methodology and Benchmark},
  author={Gaochang Wu and Yapeng Zhang and Lan Deng and Jingxin Zhang and Tianyou Chai},
  year={2024},
  page={1-1},
  DOI={10.1109/TCSVT.2024.3491865},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}
}
```
