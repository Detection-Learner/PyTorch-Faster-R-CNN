# PyTorch Faster R-CNN

This is a PyTorch version of Faster R-CNN, origin code is [here](https://github.com/rbgirshick/py-faster-rcnn)

# Install

1. Clone this repository

    ```Shell
    git clone https://github.com/Detection-Learner/PyTorch-Faster-R-CNN.git
    ```

2. Choose your `-arch` option to match your GPU.

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | Titan Black | sm_35 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |

3. Run `tools/make.sh` script

    ```Shell
    sh tools/make.sh "your GPU architecture"
    ```

    Note: default arch is "sm_35".
