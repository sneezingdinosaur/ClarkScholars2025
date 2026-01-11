# Feature-Based Multi-Modal Machine Learning for Diabetes Classification in AI-READI

1. Clone the repository:
```bash
git clone https://github.com/sneezingdinosaur/ClarkScholars2025.git
cd DiaMetrics
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```


### Training Individual Modalities

```bash
# Cardiac data
python TrainDiscrete/Cardiac/CardiacCV.py

# Clinical data
python TrainDiscrete/ClinicalData/MeasurementCVNoHB.py

# Glucose CGM data
python TrainDiscrete/GlucoseCGM/CGMCV.py

# Wearable data
python TrainDiscrete/Wearable/WearableCV.py
```

### Multimodal Training

```bash
# Early fusion
python Multimodal/earlyfusion.py

# Early fusion with health behavior
python Multimodal/earlyfusionwithHB.py

# Late fusion
python Multimodal/latefusion.py

# Late fusion with health behavior
python Multimodal/latefusionwithHB.py
```

## License

MIT License

## Acknowledgments

- Research conducted as part of Clark Scholars Program 2025
- Models trained on AIREADI health data

