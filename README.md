# Stegananalysis Suite

A open source comprehensive toolkit for detecting and analyzing steganographic content in digital files. Includes both GUI and command-line interfaces for various analysis methods. Fork it and update it in you version and way .

## What it does

This suite helps detect hidden data embedded in images and other files using multiple steganography analysis techniques. Whether you're doing digital forensics, security research, or CTF challenges, this tool provides the methods you need.

## Features

- **GUI Interface** - Easy-to-use graphical application
- **Command Line Tools** - For batch processing and automation
- **Multiple Detection Methods** - Various algorithms for different types of steganography
- **File Format Support** - Works with images, audio, and other file types
- **Analysis Reports** - Detailed results and statistics
- **Batch Processing** - Analyze multiple files at once

## Installation

```bash
git clone https://github.com/L0rdDarkk/Steganlysis-Suite.git
cd Steganlysis-Suite
pip install -r requirements.txt
```

## Usage

### GUI Mode
```bash
python gui.py
```
- Drag and drop files
- Select analysis methods
- View results visually

### Command Line Mode
```bash
python analyze.py -f image.png -m lsb
python analyze.py -d folder/ -m all
```

### Available Methods
- LSB Analysis
- Chi-Square Attack
- Statistical Analysis
- Entropy Analysis
- And more...

## Supported Files

- Images: PNG, JPEG, BMP, GIF
- Documents: PDF, TXT
- Archives: ZIP, RAR

## Use Cases

- Digital forensics investigation
- Security testing
- CTF competition solving
- Academic research
- Malware analysis

## Authors

- **Juled Mardodaj** [@L0rdDarkk](https://github.com/L0rdDarkk)
- **Bekim Fetaji** - Mentor


## Disclaimer

This tool is for educational, research, and legitimate security purposes only. Users are responsible for compliance with applicable laws and regulations.
