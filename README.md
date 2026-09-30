# SmartCity
Description      Developed an AI framework that uses DeepLabV3+ (with a ResNet50 backbone) and CNNs to perform pixel-level semantic segmentation and analyze urban street-level images.    Processed municipal records, maintenance history logs, and cityscape imagery to diagnose early signs of asset degradation and predict infrastructure service needs.
# SmartCity: AI-Based Infrastructure Analysis Using Deep Learning

An AI system that analyzes street-level city images and produces a **Smart City Score (0-100)**, a breakdown of the urban scene, detected infrastructure issues, and improvement suggestions. It uses **semantic segmentation (DeepLabV3+ with a ResNet50 backbone)** to measure how much of an image is road, sidewalk, vegetation, buildings, pedestrians and vehicles, then applies rule-based scoring on those ratios.

---

## Screenshots

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/e780b279-f401-4d7d-a910-fdcea3e9df94" />

| Dashboard | Results |
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/64f6c111-35db-4357-872a-f4fec7b8fd8d" />


| Issues and suggestions | Analysis history |
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/3089d352-f7b9-420a-925e-cc79d49c3a98" />


---

## Features

- **Semantic segmentation:** pixel-level classification of urban scenes across 19 categories.
- **Smart scoring:** weighted score (0-100) combining road quality, green coverage, pedestrian safety and traffic efficiency.
- **Issue detection:** a rule-based engine flags problems such as missing sidewalks, low vegetation, over-urbanization and low sky visibility, each with a severity (Critical, High, Medium, Low).
- **AI suggestions:** improvement recommendations generated from the segmentation ratios of each image.
- **Risk classification:** areas are labelled Low, Medium or High risk from the score and issue severity.
- **Dashboard:** login, location details form, image upload, results page with charts, and an analysis history page.

## How It Works

```
Image Input -> Segmentation -> Feature Extraction -> Rule Engine -> Smart Score
```

1. **Image input:** the user enters location details and uploads a street-level image (JPG or PNG).
2. **Segmentation:** DeepLabV3+ (ResNet50 backbone) labels every pixel using 19 Cityscapes classes.
3. **Feature extraction:** pixel ratios are computed for road, sidewalk, vegetation, buildings, pedestrians, vehicles and sky.
4. **Rule engine:** the ratios are compared with urban planning thresholds to detect issues and suggest fixes.
5. **Smart score:** a weighted score and grade are produced and stored in the analysis history.

The score is derived from the segmentation ratios, not from a separate classification output.

## Tech Stack

- **Language:** Python
- **Deep learning:** PyTorch, TorchVision (DeepLabV3+ with ResNet50)
- **Data and numerics:** NumPy, Pandas
- **Visualization:** Matplotlib
- **Dashboard:** Streamlit
- **Training data:** [Cityscapes](https://www.cityscapes-dataset.com/) (2,975 training images, 19 classes)

## Project Structure

<!-- Update this to match your actual folders and files -->

```
smartcity/
├── app.py                 # Dashboard entry point
├── models/                # Model definition and inference code
├── utils/                 # Preprocessing, ratio extraction, scoring rules
├── sample_images/         # A few example input images
├── screenshots/           # Images used in this README
├── requirements.txt
└── README.md
```

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/smartcity.git
cd smartcity

# 2. (Optional) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

## Model Weights

The trained weights file is too large to store in the repository code.
Download it from the **[Releases page](https://github.com/YOUR_USERNAME/smartcity/releases)** and place it in the `models/` folder:

```
models/your_weights_file.pth
```

## Usage

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`), sign in, fill in the location details, upload a cityscape image and click **Analyze Cityscape**.

## Dataset

This project was trained on the **Cityscapes** dataset. The dataset is not included in this repository because of its license. To retrain or evaluate the model, register and download it from the [official Cityscapes website](https://www.cityscapes-dataset.com/). A few sample images are provided in `sample_images/` for testing the app.

## Smart Score and Risk Levels

| Risk level | Meaning |
|---|---|
| Low | Urban area is performing well with balanced infrastructure |
| Medium | Urban area needs monitoring and moderate improvements |
| High | Urban area needs immediate infrastructure support and corrective action |

## Limitations

- Results depend on the quality and clarity of the input image.
- The model is trained and tested on Cityscapes, so performance may vary on other cities and camera types.
- The system uses a limited set of urban features for scoring.
- No real-time traffic or sensor data is used.
- The rule-based suggestions may need tuning for different city conditions.

## Future Work

- Real-time traffic analysis from live camera feeds
- Integration with municipal dashboards and decision-support systems
- Satellite and aerial imagery support
- Temporal comparison to track city development over time
- Multi-city benchmarking of Smart Scores

## License
<img width="896" height="1280" alt="image" src="https://github.com/user-attachments/assets/d1c3972f-992f-4491-b529-02e4624a439a" />


**Y. Harsha Vardhan Reddy**

Licensed by the 7th International Conference on
Communication and Intelligent Systems
(ICCIS 2025)
Organised By
BITS Pilani, K K Birla Goa Campus, India
