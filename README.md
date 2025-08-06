# Bike Sharing Demand Prediction

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Status](https://img.shields.io/badge/status-in%20progress-yellow.svg)

> **Note:** This project is a work in progress. The features and architecture described below represent the intended final state of the project. It is not yet complete.

This project predicts the demand for bike sharing based on various features like weather, season, and time. The model is built using a machine learning pipeline that includes data ingestion, validation, preprocessing, training, and evaluation.

## Features

*   **Data Ingestion**: Fetches data from a specified source (e.g., Google Drive).
*   **Data Validation**: Checks for missing values, duplicates, and expected columns.
*   **Data Preprocessing**: Cleans, engineers, and scales features for model training.
*   **Model Training**: Trains a machine learning model to predict bike sharing demand.
*   **Model Evaluation**: Evaluates the model's performance using various metrics.
*   **DVC Pipeline**: Uses DVC to create a reproducible machine learning pipeline.
*   **MLflow Integration**: Logs experiments, models, and metrics using MLflow.
*   **Streamlit App**: A web-based application for interacting with the model.
*   **Docker Deployment**: The application is containerized using Docker for easy deployment.
*   **CI/CD with GitHub Actions**: A CI/CD pipeline is set up with GitHub Actions to automate the deployment process.

## Architecture

### High-Level Architecture

```mermaid
graph TD
    A[GitHub] --&gt;|CI/CD| B(AWS ECR)
    B --&gt; C{AWS ECS}
    C --&gt; D[Streamlit App]
    D --&gt; E{ML Model}
```

### Low-Level Architecture

```mermaid
graph TD
    subgraph "GitHub Actions CI/CD"
        A[Push to main] --&gt; B{Build Docker Image}
        B --&gt; C{Push to AWS ECR}
        C --&gt; D{Deploy to AWS ECS}
    end

    subgraph "AWS"
        subgraph "VPC"
            E[AWS ECS] --&gt; F[Docker Container]
            F --&gt; G[Streamlit App]
            G --&gt; H{ML Model}
        end
    end

    subgraph "DVC Pipeline"
        I[Data Ingestion] --&gt; J[Data Validation]
        J --&gt; K[Data Preprocessing]
        K --&gt; L[Model Training]
        L --&gt; M[Model Evaluation]
    end

    H --&gt; M
```

## Deployment

The application is deployed to AWS using a CI/CD pipeline with GitHub Actions. When code is pushed to the `main` branch, the following steps are executed:

1.  **Build Docker Image**: A Docker image is built containing the Streamlit application and the trained model.
2.  **Push to AWS ECR**: The Docker image is pushed to a private repository in Amazon Elastic Container Registry (ECR).
3.  **Deploy to AWS ECS**: The application is deployed to Amazon Elastic Container Service (ECS) as a Fargate task, which runs the Docker container.

## Roadmap

*   [ ] Implement user authentication for the Streamlit app.
*   [ ] Add more advanced feature engineering techniques.
*   [ ] Experiment with different machine learning models.
*   [ ] Create a more detailed dashboard for model evaluation.

## Getting Started

### Prerequisites

*   Python 3.8 or higher
*   DVC
*   MLflow
*   Docker

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/Bike_Sharing_Demand_Prediction.git
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Run the DVC pipeline:
    ```bash
    dvc repro
    ```
2.  Build and run the Docker container:
    ```bash
    docker build -t bike-sharing-app .
    docker run -p 8501:8501 bike-sharing-app
    ```
3.  Open your browser and navigate to `http://localhost:8501` to use the Streamlit app.

## Project Structure

```
.
├── artifacts
│   ├── data
│   ├── models
│   └── pipelines
├── data
│   ├── interim
│   ├── processed
│   └── raw
├── logs
├── mlruns
├── S3
├── src
│   ├── components
│   │   ├── data_ingestion.py
│   │   ├── data_preprocessing.py
│   │   ├── data_validation.py
│   │   ├── feature_cleaning.py
│   │   ├── feature_engineering.py
│   │   ├── feature_scaling.py
│   │   ├── model_evaluation.py
│   │   └── model_training.py
│   └── utils
│       ├── config.py
│       ├── data_fetcher.py
│       ├── data_loader.py
│       ├── data_saver.py
│       ├── data_splitter.py
│       └── logger.py
├── .dvc
├── .dockerignore
├── .dvcignore
├── .gitattributes
├── .gitignore
├── data.dvc
├── dvc.yaml
├── Dockerfile
├── params.yaml
├── pyproject.toml
├── README.md
├── requirements.txt
└── streamlit_app.py
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the MIT License.
