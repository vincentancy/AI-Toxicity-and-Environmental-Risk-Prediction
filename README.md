# 🌍 AI-Based Chemical Toxicity & Environmental Risk Prediction

An end-to-end machine learning application for predicting the **toxicity of chemical compounds from their molecular structure** and providing an interpretable **environmental risk assessment for air, water, and soil**.

The project combines **cheminformatics, machine learning, and an interactive Streamlit interface** to provide rapid computational screening of chemical compounds from SMILES representations.

---

## 📌 Overview

Evaluating the potential toxicity of chemical compounds is important in environmental safety, drug discovery, and chemical risk assessment. Conventional experimental testing can require considerable time, resources, and laboratory work.

This project explores a computational alternative using machine learning.

A user provides the **SMILES representation of a chemical compound**, and the system:

1. Validates the molecular structure
2. Generates molecular features using RDKit
3. Predicts the probability of toxicity using a trained ML model
4. Classifies the compound as **Toxic or Non-Toxic**
5. Calculates physicochemical properties
6. Estimates **Air, Water, and Soil environmental risk**
7. Displays the results through an interactive web application

---

## ✨ Key Features

- 🧪 SMILES-based chemical input
- 🔬 Molecular structure processing with RDKit
- 🧬 Molecular fingerprint generation
- 🤖 Machine learning-based toxicity classification
- 📊 Toxicity probability prediction
- ☣️ Toxic / Non-Toxic classification
- 🌬️ Air risk assessment
- 💧 Water risk assessment
- 🌱 Soil risk assessment
- 📈 Interactive risk visualization
- 🕘 Prediction history
- 💻 Streamlit-based user interface

---

## 🔄 System Workflow

```text
Chemical SMILES
      │
      ▼
SMILES Validation
      │
      ▼
Molecular Feature Extraction
      │
      ├──────────────► Molecular Fingerprints
      │
      └──────────────► Physicochemical Descriptors
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
      Toxicity Prediction             Environmental
       using ML Model                Risk Assessment
              │                               │
              ▼                    ┌──────────┼──────────┐
     Toxicity Probability          ▼          ▼          ▼
              │                   Air       Water       Soil
              ▼                   Risk       Risk       Risk
       Toxic / Non-Toxic           │          │          │
              │                    └──────────┼──────────┘
              └───────────────────────────────┘
                              │
                              ▼
                     Streamlit Dashboard
```

---

## 📊 Dataset

The toxicity prediction component is developed using the **Tox21 (Toxicology in the 21st Century)** dataset.

Tox21 provides chemical structures along with biological activity/toxicity endpoint information that can be used to develop computational toxicity prediction models.

### Data Preparation

The machine learning pipeline includes:

- Removal of duplicate records
- Handling of missing values
- Validation of SMILES structures
- Removal of invalid molecular structures
- Processing of toxicity endpoint labels
- Preparation of a binary toxicity target
- Molecular feature generation
- Training, validation, and testing of classification models

---

## 🧬 Molecular Representation

Machine learning algorithms cannot directly interpret a SMILES string such as:

```text
CCO
```

Therefore, molecular structures are transformed into numerical representations using **RDKit**.

### Morgan Fingerprints

Morgan fingerprints encode local atomic environments and molecular substructures into a fixed-length numerical representation.

They allow machine learning models to identify structural patterns associated with toxicity.

### RDKit Fingerprints

RDKit fingerprints provide an alternative structural representation based on molecular paths and fragments.

Both fingerprint approaches can be evaluated to study how molecular representation affects model performance.

---

## 🤖 Machine Learning

The project evaluates multiple classification algorithms rather than relying on a single model.

Models explored include:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- XGBoost
- LightGBM
- Extra Trees

The models are evaluated using standard classification metrics such as:

| Metric | Purpose |
|---|---|
| **Accuracy** | Overall classification performance |
| **Precision** | Reliability of positive toxicity predictions |
| **Recall** | Ability to identify toxic compounds |
| **F1-Score** | Balance between precision and recall |
| **ROC-AUC** | Ability to distinguish between classes |

The selected trained model is serialized and integrated into the Streamlit application for real-time inference.

---

## ☣️ Toxicity Prediction Pipeline

For a new chemical compound, the prediction process follows:

```text
Input SMILES
     ↓
RDKit Molecule
     ↓
Molecular Fingerprint
     ↓
Trained ML Model
     ↓
Toxicity Probability
     ↓
Toxic / Non-Toxic
```

This allows previously unseen chemical structures to be computationally screened using the learned relationship between molecular features and toxicity labels.

---

## 🌱 Environmental Risk Assessment

Toxicity prediction and environmental risk are treated as related but separate components of the system.

While the machine learning model predicts **toxicity**, environmental risk is estimated using physicochemical properties extracted from the molecule.

Important descriptors include:

- **Molecular Weight (MW)**
- **LogP**
- **Topological Polar Surface Area (TPSA)**
- Hydrogen-bond properties
- Other structural characteristics where applicable

These properties are used in rule-based scoring to provide environmental risk indicators.

### 🌬️ Air Risk

Provides an estimate of potential risk associated with the compound's molecular characteristics relevant to air exposure or environmental behavior.

### 💧 Water Risk

Evaluates molecular characteristics associated with behavior in aqueous environments.

### 🌱 Soil Risk

Evaluates properties that may influence the compound's potential interaction with soil environments.

Risk results are presented in interpretable categories such as:

```text
LOW | MEDIUM | HIGH
```

> **Important:** Air, Water, and Soil results are rule-based screening indicators and are not direct measurements of environmental concentration, persistence, transport, or exposure.

---

## 💻 Web Application

The trained model and environmental risk assessment pipeline are integrated into an interactive **Streamlit application**.

The interface allows users to:

- Enter a SMILES string
- Run chemical analysis
- View toxicity probability
- View predicted toxicity class
- Examine Air, Water, and Soil risk
- Visualize environmental risk scores
- Review prediction history

This provides a simple interface for interacting with the underlying cheminformatics and machine learning pipeline.

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Programming | Python |
| Machine Learning | Scikit-learn, XGBoost, LightGBM |
| Cheminformatics | RDKit |
| Data Processing | Pandas, NumPy |
| Web Application | Streamlit |
| Visualization | Altair |
| Model Storage | Pickle |
| Version Control | Git & GitHub |

---

## 📂 Repository Structure

```text
AI-Toxicity-and-Environmental-Risk-Prediction/
│
├── .devcontainer/
│
├── app.py
│   └── Streamlit application
│
├── toxicity_xgboost_model.pkl
│   └── Trained toxicity prediction model
│
├── requirements.txt
│   └── Python dependencies
│
├── .gitignore
│
└── README.md
```

---
## 🚀 Live Demo

Try the deployed application here:

[🌐 Open Live Application](PASTE-YOUR-STREAMLIT-LINK-HERE)


## 🧪 Example

Enter a valid SMILES string such as:

```text
CCO
```

The application processes the molecular structure and returns results similar to:

```text
Toxicity Probability : 0.xx
Prediction           : TOXIC / NON-TOXIC
Toxicity Level       : LOW / MEDIUM / HIGH

Environmental Assessment
-------------------------
Air Risk             : LOW / MEDIUM / HIGH
Water Risk           : LOW / MEDIUM / HIGH
Soil Risk            : LOW / MEDIUM / HIGH

Overall Risk         : LOW / MEDIUM / HIGH
```

Actual results depend on the molecular structure supplied to the application.

---

## 💡 Why This Project?

This project demonstrates the integration of several areas of applied machine learning:

**Cheminformatics**

Converting molecular structures into machine-readable features using RDKit.

**Data Science**

Cleaning chemical datasets, processing labels, engineering features, and evaluating predictive models.

**Machine Learning**

Training and comparing classification algorithms for chemical toxicity prediction.

**Environmental Risk Screening**

Using molecular properties to provide interpretable environmental risk indicators.

**Application Development**

Deploying the complete prediction workflow through an interactive Streamlit application.

---

## ⚠️ Limitations

This project is intended as a **computational screening and academic machine learning system**.

Important limitations include:

- Predictions depend on the quality and coverage of the training data.
- Molecular fingerprints provide simplified representations of complex molecular structures.
- A compound outside the model's learned chemical space may produce less reliable predictions.
- Environmental risk scoring is rule-based and does not model all real-world environmental processes.
- Actual environmental risk also depends on factors such as concentration, exposure, degradation, transport, and environmental conditions.
- Computational predictions do not replace laboratory experiments or regulatory toxicological assessment.

---

## 🔮 Future Improvements

Future versions of the project could include:

- Graph Neural Networks (GNNs) for molecular modeling
- Explainable AI using SHAP
- Applicability-domain analysis
- Additional molecular descriptors
- Multi-endpoint toxicity prediction
- Improved class-imbalance handling
- Chemical name → SMILES conversion
- Integration with chemical databases
- Environmental persistence and bioaccumulation prediction
- More advanced environmental fate modeling
- Cloud deployment of the Streamlit application

---

## 🎯 Project Applications

This project can serve as a prototype for:

- Computational toxicology research
- Chemical safety screening
- Environmental informatics
- Cheminformatics
- Machine learning research
- Early-stage chemical prioritization

It is intended for **research and educational use**, not regulatory decision-making.

---

## 👩‍💻 Author

**Ancy Evelyne V**

MCA Graduate  
Machine Learning | Data Analytics | Python

GitHub: [vincentancy](https://github.com/vincentancy)

---

## ⚠️ Disclaimer

This repository is an academic machine learning project.

The predictions generated by the application are computational estimates and should **not** be interpreted as experimentally confirmed toxicity or environmental safety results.

Professional toxicological assessment, laboratory testing, and applicable regulatory guidance should be used for real-world chemical safety decisions.

---

## ⭐ Repository

If you find this project useful, consider giving the repository a **⭐ Star**.

Contributions, suggestions, and feedback are welcome.
