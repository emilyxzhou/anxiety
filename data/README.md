# About Dataset

The dataset in this repository includes a subset of electrocardiogram, electrodermal, torso posture and activity, wrist activity, and leg activity measures collected from a high- and low-anxiety group during a bug-phobic and social anxiety experience. Raw files of audio data are not made available due to protect the anonymity of the participants but can be made available upon request subject to ethics agreements. If you use this open-source dataset, please cite the following conference article where this dataset is originally published at:

### ACM Reference Format:
Hashini Senaratne, Levin Kuhlmann, Kirsten Ellis, Glenn Melvin, and Sharon Oviatt. 2021. A Multimodal Dataset and Evaluation for Feature Estimators of Temporal Phases of Anxiety. In Proceedings of the 2021 International Conference on Multimodal Interaction (ICMI ’21), October 18–22, 2021, Montréal, QC, Canada. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3462244.3479900

### BibTeX:
@inproceedings{10.1145/3462244.3479900,
author = {Senaratne, Hashini and Kuhlmann, Levin and Ellis, Kirsten and Melvin, Glenn and Oviatt, Sharon},
title = {A Multimodal Dataset and Evaluation for Feature Estimators of Temporal Phases of Anxiety},
year = {2021},
isbn = {9781450384810},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3462244.3479900},
doi = {10.1145/3462244.3479900},
booktitle = {Proceedings of the 2021 International Conference on Multimodal Interaction},
numpages = {10},
location = {Montréal, QC, Canada},
series = {ICMI '21}
}

## About Data collection methodology

To be detailed later.

### Description of the data

The dataset is presented in five directories aligned to five types of sensor metrics, where each of those directories is organized in the following structure. Each sensor metric-wise directory has two subdirectories: bug-box task and speaking task, relating to two anxiety experiences. Each of those directories is further divided into two sub-directories: high_anxiety_group and low_anxiety_group. Within those, multiple directories are organized considering the types of responses that the selected participant sample has demonstrated, and within those response-based directories, data streams for each participant can be found in .csv file format. Each .csv file has an epoch timestamp as the first column.

```
  -sensor_metric1/
    -bug_box_task
      -high_anxiety_group
        -avoidance_response
          -DataFile1
          -...
        -confrontation_response
          -DataFile1
          -...
        -escape_response
          -DataFile1
          -...
        -safety_behavior_response
          -DataFile1
          -...
      -low_anxiety_group
          -DataFile1
          -...
        -avoidance_response
          -DataFile1
          -...
        -confrontation_response
          -DataFile1
          -...
        -escape_response
          -DataFile1
          -...
        -safety_behavior_response
          -DataFile1
          -...
    -speaking_task
      -high_anxiety_group
        -confrontation_response
          -DataFile1
          -...
        -safety_behavior_response
          -DataFile1
          -...
      -low_anxiety_group
        -confrontation_response
          -DataFile1
          -...
        -safety_behavior_response
          -DataFile1
          -...

```

### Data formats

The data formats of .csv files for each sensor metric are as follows.

* ankle_movement_data

| epoch time stamp | a_x | a_y | a_z | w_x | w_y | w_z | roll | yaw | pitch |

a_x = acceleration along x-axis (likewise for y, z axes);
w_x = angular velocity along x axis (likewise for y, z axes)

* ankle_movement_data

| epoch time stamp | ECG reading |

* electrodermal_data

| epoch time stamp | Grove sensor reading |

* torso_posture_and_activity_data

| epoch time stamp | posture reading (in degrees) | activity level (in VMU) |

* wrist_activity_data

| epoch time stamp | a_x | a_y | a_z | w_x | w_y | w_z | roll | yaw | pitch |

In addition, demographics and anxiety-related details on participants and timestamps related to events in the study timeline are provided in ``participants_details.csv'' 

## Authors

Hashini Senaratne, Levin Kuhlmann, Kirsten Ellis, Glenn Melvin, and Sharon Oviatt

## License

This Anxiety Phase Dataset provided on this repository is published under the  Creative Commons Attribution-NonCommercial 4.0 International License. This means you can use it for research and educational purposes, but you must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use. You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.


## Acknowledgments

We would like to thank our participants, Mairead Cardamone-Breen and Lachlan James (support psychologists), and Yasura Vithana (an electronics engineer), for their immense support to the establishment of the described dataset after a Covid-19 lockdown.

