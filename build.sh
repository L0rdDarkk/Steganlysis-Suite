#!/bin/bash

# Base project directory
BASE_DIR="steganalysis-suite"

# Create directory structure
mkdir -p $BASE_DIR/{modules,gui,models/trained,datasets/images/{clean,steganographic,test},datasets/metadata,reports/{templates,generated},tools,tests,docs,logs}

# Create top-level files
touch $BASE_DIR/{README.md,requirements.txt,setup.py,main.py,config.yaml}

# Create module files
touch $BASE_DIR/modules/{__init__.py,detection.py,ml_detection.py,extraction.py,analysis.py,forensics.py,reports.py}

# Create GUI files
touch $BASE_DIR/gui/{__init__.py,main_window.py,detection_tab.py,analysis_tab.py,reports_tab.py,utils.py}

# Create models files
touch $BASE_DIR/models/{__init__.py,cnn_model.py,svm_model.py,model_utils.py}
touch $BASE_DIR/models/trained/{cnn_stego_detector.h5,svm_stego_detector.pkl,scaler.pkl}

# Create dataset sample files
touch $BASE_DIR/datasets/images/clean/{sample1.jpg,sample2.png}
touch $BASE_DIR/datasets/images/steganographic/{lsb_hidden.png,f5_hidden.jpg}
touch $BASE_DIR/datasets/images/test/{test1.jpg,test2.png}
touch $BASE_DIR/datasets/metadata/{clean_labels.json,stego_labels.json,dataset_info.json}

# Create report template and sample output files
touch $BASE_DIR/reports/templates/{forensic_template.html,summary_template.json}
touch $BASE_DIR/reports/generated/{analysis_YYYYMMDD_HHMMSS.pdf,detection_report.json}

# Create tools
touch $BASE_DIR/tools/{__init__.py,dataset_generator.py,batch_analyzer.py,model_trainer.py,benchmark.py}

# Create tests
touch $BASE_DIR/tests/{__init__.py,test_detection.py,test_ml_models.py,test_extraction.py,test_integration.py}

# Create documentation
touch $BASE_DIR/docs/{api_reference.md,user_guide.md,algorithms.md,research_paper.md}

# Create logs
touch $BASE_DIR/logs/{detection.log,training.log,error.log}

echo "Project structure for $BASE_DIR has been created successfully."

