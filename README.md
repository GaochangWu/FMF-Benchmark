# Fused Magnesium Smelting Process Benchmark

This is a cross-modal benchmark for the fused magnesium smelting process. The benchmark contains a total of 3 hours of synchronously acquired videos and three-phase alternating current data from different production batches. 

![Teaser Image](https://gaochangwu.github.io/FmFormer/images/FMF.png)
Cross-modal information is exploited to perform anomaly detection in the context of a typical industrial process, fused magnesium smelting, as illustrated in (a). The picture at the bottom left shows an anomaly region on the furnace shell, whose visual feature is difficult to detect due to interference from heavy water mist. A novel FMF Transformer (FmFormer) is proposed using synchronous acquired video and current data, to explore the internal features of each modality by self-attention and the correlation feature across modalities by cross-attention, as shown in (b).

## News!
- 11/25/2024:  Our pixel-level annotation dataset is now available. You can download it at: https://drive.google.com/file/d/12vQ_CHqKQ5TOK6i9whKO39OZO4Nssz_e/view?usp=sharing

- 11/01/2024:  We are in the process of preparing the datasets, which are currently not very convenient for research usage. If you would like to access the dataset in advance, please feel free to contact us: wugc\at mail\dot neu\dot edu\dot cn.

## Dataset Description

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
