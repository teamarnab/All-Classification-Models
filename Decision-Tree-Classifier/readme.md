# Decision Tree Classifier on Air Quality Dataset

This repository contains an implementation of a Decision Tree Classifier applied to an air quality dataset. The objective of the project is to predict air quality levels based on various environmental factors and evaluate the model's performance using accuracy score and confusion matrix.

## Dataset
The dataset used in this project consists of air quality measurements, including features such as pollutant levels, temperature, humidity, and other environmental variables. The target variable represents air quality levels, categorized into different classes.

## Project Structure
- `decision_tree_classifier.py`: Jupyter Notebook containing the data preprocessing, model implementation, and performance evaluation.
- `README.md`: Project documentation (this file).

## Steps Performed
1. **Data Preprocessing**:
   - Loaded the dataset and performed exploratory data analysis (EDA).
   - Handled missing values and outliers.
   - Scaled features for better model performance.
2. **Model Implementation**:
   - Split the dataset into training and testing sets.
   - Implemented a Decision Tree Classifier using `sklearn`.
   - Tuned hyperparameters for optimal performance.
3. **Evaluation Metrics**:
   - Calculated the accuracy score to measure the model's overall performance.
   - Generated a confusion matrix to evaluate prediction errors for each class.

## Results
- **Accuracy Score**: Achieved an accuracy of `~94%` on the test set.
- **Confusion Matrix**:
  - Visualized the confusion matrix to understand class-wise prediction performance.

## Requirements
To run this project, ensure you have the following installed:
- Python 3.8+
- Jupyter Notebook
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

You can install the dependencies using:
```bash
pip install -r requirements.txt
```

## How to Use
1. Clone the repository:
```bash
git clone https://github.com/teamarnab/All-Classification-Models/tree/main/Decision-Tree-Classifier
```
2. Navigate to the project directory:
```bash
cd air-quality-decision-tree
```
3. Open the Jupyter Notebook:
```bash
jupyter notebook air_quality_decision_tree.ipynb
```
4. Follow the steps in the notebook to reproduce the results.

## Visualization
- Plots and graphs have been included to visualize the data distribution and model performance.

## Future Work
- Incorporate additional machine learning algorithms for comparison.
- Explore feature engineering techniques to improve model accuracy.
- Deploy the model as a web application for real-time air quality predictions.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements
- The dataset was sourced from [mention the source if applicable].
- Thanks to the Scikit-learn documentation and community for useful resources.

---

Feel free to contribute to this project by raising issues or submitting pull requests!
