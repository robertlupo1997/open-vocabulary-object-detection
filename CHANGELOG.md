# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.1.0] - 2025-08-20

### Added
- **Self-healing pipeline load**: Automatic model loading with graceful failure handling
- **SAM2 path auto-discovery**: Intelligent detection of SAM2 checkpoint locations
- **Auto box-format detection**: Automatic conversion between normalized cxcywh and pixel xywh formats
- **Multiple prompt strategies**: 
  - `person`: Optimized for person detection
  - `common`: General object detection with common categories
  - `full`: Complete COCO class vocabulary
- **Production-ready evaluation**: COCO mAP computation with proper coordinate handling
- **Streamlit demo interface**: Interactive web interface for real-time detection
- **Comprehensive CI pipeline**: GitHub Actions with smoke tests and import checks
- **Robust testing suite**: Unit tests for core functionality with CI integration

### Fixed
- **Zero detections issue**: Fixed placeholder implementation in detector.py with actual Grounding DINO inference
- **COCO mAP calculation**: Resolved coordinate format conversion from normalized cxcywh to pixel xywh
- **Symlink path handling**: Corrected recursive symlink issues in data setup
- **Streamlit deprecation warnings**: Updated to use `use_container_width=True`

### Performance
- **Inference speed**: ~265-490ms per image on RTX 3070
- **Detection accuracy**: Non-zero COCO mAP after evaluation hardening
- **Throughput**: 1,000+ detections per 200-image evaluation run (person strategy)

### Documentation
- **Comprehensive README**: Setup, usage, and troubleshooting guide
- **Contributing guidelines**: Development workflow and coding standards
- **API documentation**: Detailed function and class documentation
- **Performance benchmarks**: Hardware-specific timing measurements

### Technical Details
- **Coordinate handling**: Robust auto-detection of box formats (cxcywh vs xyxy)
- **Model integration**: Seamless Grounding DINO + SAM 2 pipeline
- **Error recovery**: Graceful handling of missing models or data
- **Memory management**: Efficient GPU memory usage and cleanup
- **Reproducibility**: Locked environment files for consistent results

[v0.1.0]: https://github.com/robertlupo1997/open-vocabulary-object-detection/releases/tag/v0.1.0