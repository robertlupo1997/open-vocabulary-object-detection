# OVOD Data Card

## Dataset Description

The OVOD (Open-Vocabulary Object Detection) system leverages multiple datasets through its component models (Grounding DINO and SAM 2) and provides evaluation capabilities on standard benchmarks.

### Primary Evaluation Dataset

**COCO (Common Objects in Context)**
- **Purpose**: Primary evaluation benchmark for object detection performance
- **Version**: COCO 2017 (val2017 subset)
- **Size**: 5,000 validation images
- **Annotations**: 80 object categories with bounding boxes and segmentation masks
- **Usage**: mAP evaluation and baseline comparison

## Data Sources

### Component Model Training Data

The OVOD pipeline combines pre-trained models that were trained on the following datasets:

#### Grounding DINO Training Data

| Dataset | Purpose | Size | Description |
|---------|---------|------|-------------|
| **Objects365** | Object detection | 1.7M images | Large-scale object detection with 365 categories |
| **OpenImages** | Open vocabulary | 9M images | Diverse objects with hierarchical labels |
| **COCO** | Standard benchmark | 118K images | 80 common object categories |
| **Visual Genome** | Scene understanding | 108K images | Dense annotations with relationships |
| **RefCOCO/+/g** | Referring expressions | 142K expressions | Natural language object references |
| **Flickr30K Entities** | Text grounding | 31K images | Image-caption pairs with entity linking |

#### SAM 2 Training Data

| Dataset | Purpose | Size | Description |
|---------|---------|------|-------------|
| **SA-V** | Video segmentation | 50.9K videos | Diverse video content with mask annotations |
| **SA-1B** | Image segmentation | 11M images | Large-scale image segmentation masks |
| **DAVIS** | Video evaluation | 150 videos | High-quality video segmentation benchmark |
| **YouTubeVOS** | Video evaluation | 4,453 videos | Multi-object video segmentation |

### Evaluation and Testing Data

#### Included Test Data

**COCO val2017 Subset**
- **Purpose**: Performance evaluation and benchmarking
- **Images**: 1,000 randomly sampled validation images
- **Categories**: All 80 COCO object categories
- **Prompts**: Natural language descriptions for each category
- **Ground Truth**: Bounding boxes and segmentation masks

#### Custom Prompt Dataset

**Text Prompts for COCO Categories**
```
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, 
traffic light, fire hydrant, stop sign, parking meter, bench, bird, 
cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack,
umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball,
kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket,
bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, 
sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair,
couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, 
remote, keyboard, cell phone, microwave, oven, toaster, sink, 
refrigerator, book, clock, vase, scissors, teddy bear, hair drier, 
toothbrush
```

## Data Processing

### Preprocessing Pipeline

1. **Image Preprocessing**
   - Resize to target resolution (640px default)
   - Maintain aspect ratio with padding
   - Normalize to [0, 1] range
   - Convert to RGB format

2. **Text Preprocessing**
   - Clean and normalize prompts
   - Split multi-object descriptions
   - Expand synonyms for better recall
   - Format for Grounding DINO input

3. **Post-processing**
   - Non-Maximum Suppression (NMS)
   - Score filtering and ranking
   - Coordinate scaling to original image size
   - Mask format conversion

### Data Augmentation

**Training** (Applied to base models during original training):
- Random horizontal flipping
- Color jittering and brightness adjustment
- Random cropping and scaling
- Mosaic augmentation (for detection)

**Inference** (Applied in OVOD pipeline):
- Multi-scale testing support
- Test-time augmentation options
- Robust prompt processing

## Licensing and Usage Rights

### Dataset Licenses

| Dataset | License | Commercial Use | Restrictions |
|---------|---------|----------------|--------------|
| **COCO** | CC BY 4.0 | ✅ Yes | Attribution required |
| **Objects365** | CC BY 4.0 | ✅ Yes | Attribution required |
| **OpenImages** | CC BY 4.0 | ✅ Yes | Attribution required |
| **Visual Genome** | CC BY 4.0 | ✅ Yes | Attribution required |
| **SA-V/SA-1B** | Custom License | ⚠️ Research | Non-commercial research use |

### Usage Guidelines

- **Research Use**: All datasets support academic research
- **Commercial Use**: COCO, Objects365, OpenImages support commercial applications
- **Attribution**: Proper citation required for all datasets
- **Redistribution**: Follow individual dataset redistribution policies

## Data Quality and Characteristics

### Geographic Distribution

The training data covers global geographic regions with emphasis on:
- **North America**: 40% (primarily US/Canada)
- **Europe**: 25% (Western and Eastern Europe)  
- **Asia**: 20% (China, Japan, India, Southeast Asia)
- **Other Regions**: 15% (South America, Africa, Oceania)

### Temporal Coverage

- **Collection Period**: 2010-2023
- **Image Content**: Contemporary objects and scenes
- **Seasonal Variation**: All seasons represented
- **Time of Day**: Various lighting conditions

### Domain Diversity

#### Object Categories
- **People**: Various ages, ethnicities, clothing styles
- **Vehicles**: Cars, trucks, motorcycles, bicycles, aircraft
- **Animals**: Domestic pets, livestock, wildlife
- **Indoor Objects**: Furniture, electronics, kitchen items
- **Outdoor Objects**: Street furniture, natural elements

#### Scene Types
- **Indoor**: Homes, offices, restaurants, shops
- **Outdoor**: Streets, parks, beaches, mountains
- **Urban**: City scenes, traffic, buildings
- **Rural**: Countryside, farms, natural landscapes

## Bias and Limitations

### Known Biases

1. **Geographic Bias**
   - Over-representation of Western countries
   - Under-representation of developing regions
   - Urban scenes more common than rural

2. **Demographic Bias**
   - Age distribution skewed toward adults
   - Potential gender representation imbalances
   - Socioeconomic status reflected in object types

3. **Linguistic Bias**
   - Training primarily in English
   - Western cultural context in descriptions
   - Limited multilingual text grounding

### Quality Limitations

1. **Annotation Quality**
   - Varying annotation standards across datasets
   - Potential labeling errors or inconsistencies
   - Subjective interpretation in complex scenes

2. **Coverage Gaps**
   - Rare or emerging object categories
   - Fine-grained distinctions within categories
   - Context-dependent object descriptions

3. **Technical Limitations**
   - Image resolution constraints
   - Compression artifacts in web-sourced images
   - Temporal bias toward recent years

## Ethical Considerations

### Privacy Concerns

- **Human Subjects**: People appear in training images
- **Consent**: Original dataset consent policies apply
- **Sensitive Content**: Potential for private/personal objects
- **Recommendation**: Avoid using for surveillance or identification

### Fairness and Representation

- **Demographic Representation**: Monitor for biased performance across groups
- **Cultural Sensitivity**: Be aware of cultural context in object descriptions  
- **Accessibility**: Consider diverse user needs and capabilities
- **Mitigation**: Regular bias auditing and inclusive evaluation

### Responsible Use

1. **Intended Applications**
   - ✅ Research and education
   - ✅ Accessibility tools
   - ✅ Content organization
   - ✅ Interactive applications

2. **Discouraged Applications**
   - ❌ Mass surveillance
   - ❌ Discriminatory profiling
   - ❌ Privacy-invasive monitoring
   - ❌ Critical safety systems without validation

## Data Maintenance

### Version Control

- **Current Version**: v1.0
- **Last Updated**: 2024-08-19
- **Update Frequency**: As base models are updated
- **Changelog**: Track modifications and improvements

### Quality Assurance

- **Validation**: Regular evaluation on held-out data
- **Monitoring**: Performance tracking across demographics
- **Feedback**: Community reporting of issues
- **Improvement**: Continuous refinement based on feedback

### Data Governance

- **Access Control**: Public availability with proper attribution
- **Usage Tracking**: Monitor applications and use cases
- **Compliance**: Adhere to dataset licensing requirements
- **Documentation**: Maintain comprehensive data provenance

## Contact and Support

### Data Questions

For questions about data usage, licensing, or quality:
- **Email**: your-email@domain.com
- **Issues**: [GitHub Issues](https://github.com/your-repo/ovod/issues)
- **Documentation**: [Project Documentation](https://your-repo.github.io/ovod)

### Reporting Issues

To report data quality issues or ethical concerns:
1. Open a GitHub issue with detailed description
2. Include specific examples or evidence
3. Suggest potential mitigations or improvements
4. Follow community guidelines for constructive feedback

---

*Data card last updated: 2024-08-19*