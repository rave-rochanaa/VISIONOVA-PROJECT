VisioNova 📊
A streamlined, no-code data visualization and insight generation tool that empowers users to upload datasets, explore interactive charts, run built-in machine learning models, and extract meaningful narratives — all through an intuitive web interface.
✨ Key Features

🎨 Interactive Chart Builder: Create bar, line, scatter, pie, box, and heatmap charts with ease
🤖 Built-in ML Models: Regression & classification models powered by scikit-learn
💡 Smart Chart Recommendations: Get chart suggestions based on your data structure
🗣️ Visual Q&A Engine: Ask questions like "Show ROI trend" and get instant visualizations
📝 Auto-generated Insights: Automatic insight summaries and performance alerts
📂 Local File Support: Upload CSV/XLSX files with in-app preprocessing
📤 Export Options: Save as PDF, PNG, or CSV with no database or cloud requirements
🔒 Privacy-First: Fully local, offline-capable with no API dependencies

🚀 Quick Start
Prerequisites

Python 3.7 or higher
pip package manager

Installation

Clone the repository
bashgit clone https://github.com/your-username/visio-nova.git
cd visio-nova

Install dependencies
bashpip install -r requirements.txt

Run the application
bashstreamlit run app.py

Open your browser and navigate to http://localhost:8501
Upload your data and start exploring! 🎉

🛠️ Tech Stack
ComponentTechnologyFrontendStreamlit, Plotly, SeabornBackendPython, Pandas, NumPyMachine Learningscikit-learnArchitectureFully local, offline-capable
🧠 Machine Learning Capabilities
Regression Models

Linear Regression
Ridge Regression
Random Forest Regressor

Classification Models

Logistic Regression
Decision Tree Classifier

Analysis Tools

Outlier detection
Correlation matrix analysis
Trend and seasonality detection

All ML outputs are visualized and explained with auto-generated narratives to make insights accessible to everyone.
📋 Requirements
Create a requirements.txt file with the following dependencies:
txtstreamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.15.0
seaborn>=0.12.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
openpyxl>=3.1.0
🖥️ Usage Examples
Basic Data Exploration

Upload your CSV or XLSX file
Preview your data structure
Select columns for analysis
Choose visualization type or let VisioNova recommend

Machine Learning Workflow

Select your target variable
Choose features for modeling
Pick a suitable algorithm
View model performance metrics
Generate predictions and insights

Export and Share

Generate your visualizations
Export as PDF report or PNG images
Download processed data as CSV
Share insights with stakeholders

🎯 Use Cases

Business Analytics: Track KPIs, sales trends, and performance metrics
Research: Explore datasets and generate publication-ready charts
Education: Teach data science concepts with interactive examples
Personal Projects: Analyze personal data like fitness, finance, or hobbies

🤝 Contributing
We welcome contributions! Here's how you can help:

Fork the repository
Create a feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

Development Setup
bash# Clone your fork
git clone https://github.com/your-username/visio-nova.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If you have dev dependencies

# Run tests
pytest tests/
