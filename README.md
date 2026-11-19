# Sentiment Analysis of Car Dealership and Board Game Reviews

This repository accompanies Kimon Andreou’s 2024 doctoral dissertation at Nova Southeastern University. The study uses review ratings to label sentiment as positive, neutral, or negative. It compares a naïve Bayes baseline with convolutional neural network (CNN), bidirectional long short-term memory (BiLSTM), and hybrid CNN-BiLSTM models across two review domains.

## Dissertation

Andreou, K. (2024). *Enhancing Sentiment Analysis in Niche Domains: Introducing Diverse Datasets and Evaluating Model Performance in Car Dealership and Board Game Reviews* [Doctoral dissertation, Nova Southeastern University](https://nsuworks.nova.edu/gscis_etd/1201/). NSUWorks.

## Datasets and analysis notebooks

Download the JSON datasets from the [`data-v1` release](https://github.com/kimon1230/nova_doctoral/releases/tag/data-v1). Open either report notebook on GitHub to view its saved results.

| Reviews | Dataset | Report notebook | Python loader |
| --- | --- | --- | --- |
| Trustpilot car dealerships | [`tp_data.json`](https://github.com/kimon1230/nova_doctoral/releases/download/data-v1/tp_data.json) (273 MB) | [`TP_Report_final.ipynb`](./TP_Report_final.ipynb) | [`tp.py`](./tp.py) |
| BoardGameGeek’s top 250 board games | [`bgg_data.json`](https://github.com/kimon1230/nova_doctoral/releases/download/data-v1/bgg_data.json) (1.15 GB) | [`BGG_Report_Final.ipynb`](./BGG_Report_Final.ipynb) | [`bgg.py`](./bgg.py) |

## Rerun the analysis

1. Clone the repository.

   ```sh
   git clone https://github.com/kimon1230/nova_doctoral.git
   cd nova_doctoral
   ```

2. Download the JSON file for the report you want to run and place it in the repository folder alongside the report and its Python loader.
3. Open the report notebook. In Google Colab, make the folder available in Google Drive and update the working directory in the Drive setup cell. In local Jupyter, replace that cell with a change to the repository folder. Review the dependency cell, which specifies TensorFlow 2.17 and Keras 3.5.

The loaders read the full JSON file into memory. Allow for this when choosing a runtime, especially for the BoardGameGeek dataset.

## Data collection and preparation

| Notebook | Role |
| --- | --- |
| [`trustpilot_reviews_car_dealerships.ipynb`](./trustpilot_reviews_car_dealerships.ipynb) | Collects Trustpilot car dealership reviews. |
| [`boardgamegeek_reviews.ipynb`](./boardgamegeek_reviews.ipynb) | Collects BoardGameGeek reviews. |
| [`combine and cleanup.ipynb`](./combine%20and%20cleanup.ipynb) | Combines and cleans the collected reviews to build the datasets. |

## License

The original software and documentation are licensed under MIT. CC BY 4.0 applies to the rights Kimon A. Andreou holds in the datasets. Third-party reviews and other source material remain subject to their respective rights. [License and dataset attribution](./LICENSE.md).
